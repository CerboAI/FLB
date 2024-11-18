// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import Tokenizers

/// Layers to apply LoRA adapters to.
///
/// This is the value returned by ``LoRAModel/loraLinearLayers()``.
public typealias LoRALinearLayers = [(Module, [String])]

public protocol LoRAModel {
    /// Return the layers and keys to apply LoRA adapters to.
    ///
    /// For example this might apply the adapters to the `q` an `v` projections in the
    /// Attention layers:
    ///
    /// ```swift
    /// model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    /// ```
    ///
    /// It is not required that a model implement this protocol to have LoRA adapters applied, but
    /// the command line driver example uses this to produce the ``LoRALinearLayers``.
    ///
    /// ### See Also
    /// - ``LoRATrain/convert(model:layers:)``
    func loraLinearLayers() -> LoRALinearLayers
}

/// Protocol for LoRA implementations that provides a method for converting back to a `Linear`
/// (or subtype).
///
/// This is normally called via ``LoRATrain/fuse(model:layers:deQuantize:)``
public protocol LoRAConvertToLinear {
    func toLinear(deQuantize: Bool) -> Linear
}

/// Implementation of LoRA `Linear` replacement layer.
///
/// This layer implements the LoRA capabilities for `Linear` layers, specifically:
///
/// - converting `Linear` or `QuantizedLinear` layers to ``LoRALinear`` / ``QLoRALinear``
/// - converting ``LoRALinear`` back to `Linear` or `QuantizedLinear` (``LoRAConvertToLinear``)
/// - implementing the LoRA evaluation
///
/// ``QLoRALinear`` is the equivalent class for `QuantizedLinear`.
///
/// This is not typically used directly -- ``LoRATrain/convert(model:layers:)`` is used to
/// add the adapter layers to a given model.
///
/// ### See Also
/// - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
/// - [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
/// - ``QLoRALinear``
/// - ``LoRATrain/convert(model:layers:)``
/// - ``LoRATrain/fuse(model:layers:deQuantize:)``
public class LoRALinear: Linear, LoRAConvertToLinear {

    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray

    required public init(
        _ inputDimensions: Int, _ outputDimensions: Int, rank: Int = 8, bias: Bool = false,
        scale: Float = 20.0, linear: Linear
    ) {
        // Scale for low-rank update
        self.scale = scale

        // Low rank lora weights
        let loraScale = 1 / sqrt(Float(inputDimensions))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])

        super.init(weight: linear.weight, bias: linear.bias)

        freeze()
    }

    /// Freeze all parameters except the lora parameters
    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        // realize the keys and omit the lora parameters
        let keys =
            (keys ?? self.filterMap(filter: Self.filterLocalParameters).flattened().map { $0.0 })
            .filter {
                $0 != "lora_a" && $0 != "lora_b"
            }
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    /// Convert a `Linear` or `QuantizedLinear` layer into a new `Linear` layer
    /// that implements the `LoRA` adapter.
    ///
    /// This is typically called via ``LoRATrain/convert(model:layers:)``.
    ///
    /// ### See Also
    /// - ``LoRATrain/convert(model:layers:)``
    /// - ``QLoRALinear/from(linear:rank:)``
    public static func from(linear: Linear, rank: Int = 8) -> Linear {
        if let linear = linear as? QuantizedLinear {
            return QLoRALinear.from(linear: linear, rank: rank)
        }
        let (outputDimensions, inputDimensions) = linear.shape
        return LoRALinear(inputDimensions, outputDimensions, rank: rank, linear: linear)
    }

    /// Convert back into a fused `Linear` layer.
    ///
    /// This is typically called via ``LoRATrain/fuse(model:layers:deQuantize:)``.
    ///
    /// ### See Also
    /// - ``LoRATrain/fuse(model:layers:deQuantize:)``
    /// - ``LoRAConvertToLinear``
    /// - ``QLoRALinear/toLinear(deQuantize:)``
    public func toLinear(deQuantize: Bool = false) -> Linear {
        let dtype = weight.dtype
        let loraB = (scale * loraB.T).asType(dtype)
        let loraA = loraA.T.asType(dtype)
        return Linear(weight: weight + matmul(loraB, loraA), bias: bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x.asType(weight.dtype))
        let z = matmul(matmul(x, self.loraA), self.loraB)
        return y + scale * z
    }
}

/// Implementation of LoRA `QuantizedLinear` replacement layer.
///
/// See ``LoRALinear`` (equivalent class for `Linear` layers) for more information.
public class QLoRALinear: QuantizedLinear, LoRAConvertToLinear {

    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray

    required public init(
        _ inputDimensions: Int, _ outputDimensions: Int, rank: Int = 8, bias: Bool = false,
        scale: Float = 20.0, linear: QuantizedLinear
    ) {

        // Scale for low-rank update
        self.scale = scale

        // Low rank lora weights
        let loraScale = 1 / sqrt(Float(inputDimensions))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])

        super.init(
            weight: linear.weight, bias: linear.bias, scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits)

        // start frozen except for the lora keys
        freeze()
    }

    /// Freeze all parameters except the lora parameters
    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        // realize the keys and omit the lora parameters
        let keys =
            (keys ?? self.filterMap(filter: Self.filterLocalParameters).flattened().map { $0.0 })
            .filter {
                $0 != "lora_a" && $0 != "lora_b"
            }
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    /// Convert a `QuantizedLinear` layer into a new `Linear` layer
    /// that implements the `LoRA` adapter.
    ///
    /// This is typically called via ``LoRATrain/convert(model:layers:)``.
    ///
    /// ### See Also
    /// - ``LoRATrain/convert(model:layers:)``
    /// - ``LoRALinear/from(linear:rank:)``
    public static func from(linear: QuantizedLinear, rank: Int = 8) -> Linear {
        var (outputDimensions, inputDimensions) = linear.shape
        inputDimensions = inputDimensions * 32 / linear.bits
        return QLoRALinear(inputDimensions, outputDimensions, rank: rank, linear: linear)
    }

    /// Convert back into a fused `QuantizedLinear` layer.
    ///
    /// This is typically called via ``LoRATrain/fuse(model:layers:deQuantize:)``.
    ///
    /// ### See Also
    /// - ``LoRATrain/fuse(model:layers:deQuantize:)``
    public func toLinear(deQuantize: Bool = false) -> Linear {
        // convert back into full weights
        let weight = dequantized(
            weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits)

        let loraB = (scale * loraB.T).asType(.float16)
        let loraA = loraA.T.asType(.float16)

        // convert back into quantized
        return QuantizedLinear(
            weight: weight + matmul(loraB, loraA), bias: bias, groupSize: groupSize, bits: bits)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x.asType(scales.dtype))
        let z = matmul(matmul(x, self.loraA), self.loraB)
        return y + scale * z
    }
}

/// Equivalent to `lora.py/iterate_batches()`.  Used internally by ``LoRATrain``.
struct LoRABatchIterator: Sequence, IteratorProtocol {

    let dataset: [String]
    let batchSize: Int
    let tokenizer: Tokenizer

    let train: Bool

    var indices: [Int]
    var index = 0

    public init(dataset: [String], tokenizer: Tokenizer, batchSize: Int, train: Bool) {
        self.dataset = dataset
        self.batchSize = batchSize
        self.tokenizer = tokenizer
        self.train = train

        self.indices = Array(0 ..< dataset.count)
        if train {
            indices.shuffle()
        }
    }

    mutating public func next() -> (MLXArray, MLXArray, MLXArray)? {
        if index >= indices.count {
            if !train {
                return nil
            }

            indices.shuffle()
            index = 0
        }

        let endIndex = Swift.min(index + batchSize, indices.count)

        let batch = (index ..< endIndex)
            .map { tokenizer.encode(text: dataset[indices[$0]]) }
        let lengths = batch.map { $0.count }
        let maxLength = lengths.max() ?? 0

        if maxLength > 2048 {
            print(
                """
                [WARNING] Some sequences are longer than 2048 tokens.
                Consider pre-splitting your data to save memory.
                """)
        }

        // pad to the max length
        let batchArray = MLXArray.zeros([lengths.count, maxLength], type: Int32.self)
        for (j, (b, l)) in zip(batch, lengths).enumerated() {
            batchArray[j, 0 ..< l] = MLXArray(b)
        }

        index = endIndex

        return (batchArray[0..., .stride(to: -1)], batchArray[0..., 1...], MLXArray(lengths))
    }

}

/// Collection of functions for adding LoRA adapters to an LLM model, training, fusing and saving/loading weights.
///
/// The typical flow for training is:
///
/// ```swift
/// // load the base model and tokenizer
/// let (model, tokenizer) = try await LLM.load(configuration: ModelConfiguration.mistral7B4bit)
///
/// // add LoRALinear adapter layers
/// LoRATrain.convert(model: model, layers: Array(model.loraLinearLayers().suffix(4)))
///
/// // optionally load LoRA weights
/// try LoRATrain.loadLoRAWeights(model: model, url: ...)
///
/// // load the train/validation data
/// let train = try loadLoRAData(directory: data, name: "train")
/// let valid = try loadLoRAData(directory: data, name: "valid")
///
/// // train
/// let optimizer = Adam(learningRate: 1e-5)
/// try await LoRATrain.train(
///     model: model, train: train, validate: valid, optimizer: optimizer, tokenizer: tokenizer,
///     parameters: LoRATrain.Parameters()
/// ) { progress in
///     print(progress)
///     return .more
/// }
/// ```
///
/// At this point the model will be trained and you could do one of the following:
///
/// - ``saveLoRAWeights(model:url:)`` -- write the LoRA weights to a file
/// - ``fuse(model:layers:deQuantize:)`` -- fuse the LoRA weights and convert back into the original model
///     architecture.  These weights can be saved and reloaded with normal model handling code.
/// - ``evaluate(model:dataset:loss:tokenizer:batchSize:batchCount:)``-- compute the test loss
///     againts a test dataset
/// - use the in memory model as a normal `LLMModel` and evaluate a prompt
///
public enum LoRATrain {

    public typealias LoraLossFunction = (Module, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray
    )
    

    /// LoRA training parameters
    public struct Parameters: Sendable {
        /// number of prompts to evaluate per iteration
        public var batchSize = 4

        /// number of iterations to train for
        public var iterations = 1000

        /// number of training steps between loss reporting
        public var stepsPerReport = 10

        /// number of steps between validations
        public var stepsPerEval = 100

        /// number of validations batches, `0` uses the entire validation set
        public var validationBatches = 10

        /// save the model every N iterations
        public var saveEvery = 100

        /// save path for the adapter `.safetensors`
        public var adapterURL: URL?

        public var current_trainingLoss: Double = 0.0
//        public var currentMemoryUsage: Double = 0.0
        
        
       
        
        public init(
            batchSize: Int = 4, iterations: Int = 1000, stepsPerReport: Int = 1,
            stepsPerEval: Int = 100, validationBatches: Int = 10, saveEvery: Int = 100,
            adapterURL: URL? = nil
        ) {
            self.batchSize = batchSize
            self.iterations = iterations
            self.stepsPerReport = stepsPerReport
            self.stepsPerEval = stepsPerEval
            self.validationBatches = validationBatches
            self.saveEvery = saveEvery
            self.adapterURL = adapterURL
        }
    }

    /// Freeze the model layers and replace the indicated modules (Linear) that should be
    /// converted to ``LoRALinear`` and remain trainable.
    ///
    /// Once a model has had the LoRA adapters applied, adapter weights can be loaded
    /// (if available):
    ///
    /// ```swift
    /// try LoRATrain.loadLoRAWeights(model: model, url: args.adapter)
    /// ```
    ///
    /// At this point the model is ready for one or more of the following:
    ///
    /// - training with ``train(model:train:validate:optimizer:loss:tokenizer:parameters:progress:)``
    /// - loss evaluation with ``evaluate(model:dataset:loss:tokenizer:batchSize:batchCount:)``
    /// - fusing with ``fuse(model:layers:deQuantize:)``
    /// - text generation with ``generate(promptTokens:parameters:model:tokenizer:additionalEOSTokens:didGenerate:)``
    ///     - note that this is just using normal model text generation
    ///
    /// - Parameters:
    ///   - model: model to convert
    ///   - layers: number of suffix layers to convert
    public static func convert(model: Module, layers: LoRALinearLayers) {
        model.freeze()

        for (layer, keys) in layers {
            var update = ModuleChildren()
            let children = layer.children()
            for key in keys {
                if let item = children[key], case .value(let child) = item {
                    if let linear = child as? Linear {
                        update[key] = .value(LoRALinear.from(linear: linear))
                    } else {
                        print("\(key) on \(layer) is not Linear")
                    }
                } else {
                    print("failed to find key \(key) on \(layer)")
                }
            }
            layer.update(modules: update)
        }
    }

    /// Fuses the LoRA adapters back into the model weights.
    ///
    /// This produces a model in the original format with `Linear` or `QuantizedLinear` layer
    /// weights that incorporate the LoRA adapter.
    ///
    /// - Parameters:
    ///   - model: model to convert
    ///   - deQuantize: if `true` will convert `QuantizedLinear` back into `Linear`
    public static func fuse(model: Module, layers: LoRALinearLayers, deQuantize: Bool = false) {
        for (layer, keys) in layers {
            var update = ModuleChildren()
            let children = layer.children()
            for key in keys {
                if let item = children[key], case .value(let child) = item {
                    if let lora = child as? LoRAConvertToLinear {
                        update[key] = .value(lora.toLinear(deQuantize: deQuantize))
                    }
                }
            }
            if !update.isEmpty {
                layer.update(modules: update)
            }
        }
    }

    public static func loss(model: Module, inputs: MLXArray, targets: MLXArray, lengths: MLXArray)
        -> (
            MLXArray, MLXArray
        )
    {
        // def loss(model, inputs, targets, lengths):

        // run model on inputs
        let model = model as! LLMModel
        let logits = model(inputs, cache: nil).asType(.float32)

        // mask padding tokens
        let lengthMask = MLXArray(0 ..< inputs.dim(1))[.newAxis, 0...] .< lengths[0..., .newAxis]

        // calculate the loss
        let ntoks = lengthMask.sum()
        let ce = (crossEntropy(logits: logits, targets: targets) * lengthMask).sum() / ntoks
        return (ce, ntoks)
    }

    /// Evaluate the model and dataset and return the loss over the entire dataset.
    ///
    /// - Parameters:
    ///   - model: the model to evaluate
    ///   - dataset: the dataset
    ///   - loss: loss function
    ///   - tokenizer: tokenizer
    ///   - batchSize: number of items from the dataset to evaluate at once
    ///   - batchCount: number of batch elements to evaluate, 0 for all
    /// - Returns: the loss over the enumerate data
    ///
    /// ### See Also
    /// - ``loadLoRAData(directory:name:)``
    public static func evaluate(
        model: Module, dataset: [String], loss: LoraLossFunction = loss, tokenizer: Tokenizer,
        batchSize: Int, batchCount: Int
    ) -> Float {
        var allLosses = [Float]()
        var tokenCount = 0

        for (iteration, (inputs, targets, lengths)) in LoRABatchIterator(
            dataset: dataset, tokenizer: tokenizer, batchSize: batchSize, train: false
        ).enumerated() {
            let (losses, tokens) = loss(model, inputs, targets, lengths)
            allLosses.append((losses * tokens).item(Float.self))
            tokenCount += tokens.item(Int.self)

            if batchCount != 0 && iteration + 1 >= batchCount {
                break
            }
        }

        return (sum(MLXArray(allLosses), stream: .cpu) / tokenCount).item(Float.self)
    }

    /// Given a model with LoRA adaptors applied, load adapter weights from a `.safetensors` file.
    ///
    /// ### See Also
    /// - ``convert(model:layers:)``
    /// - ``saveLoRAWeights(model:url:)``
    public static func loadLoRAWeights(model: Module, url: URL) throws {
        let weights = try ModuleParameters.unflattened(loadArrays(url: url))
        try model.update(parameters: weights, verify: .noUnusedKeys)
        eval(model)
    }

    /// Given a model with LoRA adaptors applied, write adapter weights to a `.safetensors` file.
    ///
    /// ### See Also
    /// - ``convert(model:layers:)``
    /// - ``loadLoRAWeights(model:url:)``
    public static func saveLoRAWeights(model: Module, url: URL) throws {
        let parameters = Dictionary(
            uniqueKeysWithValues: model.trainableParameters().flattened())
        try save(arrays: parameters, url: url)
        
        
        // ####################################################################################################################
//        let selectedKeys = ["model.layers.29.self_attn.q_proj.lora_a", "model.layers.29.self_attn.v_proj.lora_b"]
//        let selectedKeys = parameters.keys.first
        let selectedKeys = [parameters.keys.first?.description]
        let filteredParameters = parameters.filter { selectedKeys.contains($0.key) }

 
        var asciiStrings: [String] = []
        for (key, value) in filteredParameters {
            for i in 0...7 {
                let temp = value[0][i]
                
                let scalarValue: Float = temp.item()  // 直接提取数值
                let floatString = String(format: "%.6f", scalarValue)  // 保留 6 位小数
                if let asciiData = floatString.data(using: .ascii) {
                    print("ASCII Data: \(asciiData)")
                    if let asciiString = String(data: asciiData, encoding: .ascii) {
                        print("ASCII String: \(asciiString)")
                        asciiStrings.append(asciiString)
                    }
                }
            }
 
            print(asciiStrings)
            saveASCIIStringsToFile(asciiStrings)
            
            if let asciiStrings = readASCIIStringsFromFile() {
                print("Read ASCII Strings:", asciiStrings)
            } else {
                print("Failed to read ASCII strings from file.")
            }
            
            
            break
        }

        

 
        // ##################################################################################################

    }

    public enum Progress: CustomStringConvertible, Sendable {
        case train(
            iteration: Int, trainingLoss: Float, iterationsPerSecond: Double,
            tokensPerSecond: Double)
        case validation(iteration: Int, validationLoss: Float, validationTime: Double)
        case save(iteration: Int, url: URL)

        public var description: String {
            switch self {
            case .train(
                let iteration, let trainingLoss, let iterationsPerSecond, let tokensPerSecond):
                "Iteration \(iteration + 1): training loss \(trainingLoss.formatted()), "
                    + "iterations/sec \(iterationsPerSecond.formatted()), "
                    + "Tokens/sec \(tokensPerSecond.formatted())"
                
 
            case .validation(let iteration, let validationLoss, let validationTime):
                "Iteration \(iteration + 1): "
                    + "validation loss \(validationLoss.formatted()), "
                    + "validation time \(validationTime.formatted())s"
            case .save(let iteration, let url):
                "Iteration \(iteration + 1): saved weights to \(url.path())"
            }
        }
    }

    public enum ProgressDisposition: Sendable {
        case stop
        case more
    }

    /// Train (or continue training) LoRA weights.
    ///
    /// - Parameters:
    ///   - model: model to train
    ///   - train: training dataset
    ///   - validate: validate dataset
    ///   - optimizer: optimizer used in training
    ///   - loss: loss function
    ///   - tokenizer: tokenizer
    ///   - parameters: training parameters
    ///   - progress: progress callback
    public static func train(
        model: Module, train: [String], validate: [String], optimizer: Optimizer,
        loss: @escaping LoraLossFunction = loss, tokenizer: Tokenizer, parameters: Parameters,
        progress: (Progress) -> ProgressDisposition
    ) throws {
        // def train(model, train_set, val_set, optimizer, loss, tokenizer, args)

        let lossValueGrad = valueAndGrad(model: model) { model, arrays in
            let (ce, ntoks) = loss(model, arrays[0], arrays[1], arrays[2])
            return [ce, ntoks]
        }

        var losses = [Float]()
        var tokenCount = 0

        var start = Date.timeIntervalSinceReferenceDate

        
        
        
        for (iteration, (inputs, targets, lengths)) in LoRABatchIterator(
            dataset: train, tokenizer: tokenizer, batchSize: parameters.batchSize, train: true
        ).enumerated() {
            // forward and backward pass
            let (resultArray, grad) = lossValueGrad(model, [inputs, targets, lengths])
            let lvalue = resultArray[0]
            let tokens = resultArray[1]

            // model update
            optimizer.update(model: model, gradients: grad)
            eval(model, optimizer, lvalue)

            // record loss
            losses.append(lvalue.item(Float.self))
            tokenCount += tokens.item(Int.self)
//
            
            
            
            
            
            
            
            
            
            // report training loss
            if (iteration + 1) % parameters.stepsPerReport == 0 {
                let trainingLoss = MLXArray(losses).mean(stream: .cpu).item(Float.self)
                let now = Date.timeIntervalSinceReferenceDate

                let iterationsPerSecond = Double(parameters.stepsPerReport) / (now - start)
                let tokensPerSecond = Double(tokenCount) / (now - start)
                
                // 保存 trainingLoss 到文件
                saveTrainingLossToFile(trainingLoss)
                
                // ############################################
                if let content = readTrainingLossFromFile() {
                    if let mean = calculateMean(from: content) {
                        print("Mean value: \(mean)")
                    }
//                    print("File content:\n\(content)")
                } else {
                    print("Failed to read file content.")
                }
                // ############################################
                
                
                

                if progress(
                    .train(
                        iteration: iteration, trainingLoss: trainingLoss,
                        iterationsPerSecond: iterationsPerSecond, tokensPerSecond: tokensPerSecond))
                    == .stop
                {
                    break
                }

                losses.removeAll()
                tokenCount = 0
                start = Date.timeIntervalSinceReferenceDate
            }
            
            
            
            
            
            
            
            
            
            

            // report validation loss
            if iteration == 0 || (iteration + 1) % parameters.stepsPerEval == 0 {
                let validationStart = Date.timeIntervalSinceReferenceDate
                let validationLoss = evaluate(
                    model: model, dataset: validate, loss: loss, tokenizer: tokenizer,
                    batchSize: parameters.batchSize, batchCount: parameters.validationBatches)
                let now = Date.timeIntervalSinceReferenceDate

                if progress(
                    .validation(
                        iteration: iteration, validationLoss: validationLoss,
                        validationTime: now - validationStart)) == .stop
                {
                    break
                }

                start = Date.timeIntervalSinceReferenceDate
            }

            
            // ############################################
 
            if let adapterURL = createAdapterURL(fileName: "lora_weights"), (iteration + 1) % parameters.saveEvery == 0 {
                print("Adapter save path: \(adapterURL)")
                
                try saveLoRAWeights(model: model, url: adapterURL)
                
                if progress(.save(iteration: iteration, url: adapterURL)) == .stop {
                    break
                }
                
                start = Date.timeIntervalSinceReferenceDate
                
                
                
                if let fileSize = getFileSize(at: adapterURL) {
//
                    let fileSizeKB = Double(fileSize) / 1024.0
                    print("File size: \(fileSizeKB) KB")
                    
                } else {
                    print("Failed to retrieve file size.")
                }
            }
            
 

            // ############################################
            
            
//            // save adapter weights if needed
//            if let adapterURL = parameters.adapterURL, (iteration + 1) % parameters.saveEvery == 0 {
//                try saveLoRAWeights(model: model, url: adapterURL)
//
//                if progress(.save(iteration: iteration, url: adapterURL)) == .stop {
//                    break
//                }
//
//                start = Date.timeIntervalSinceReferenceDate
//            }
//            
            
            

            if iteration + 1 >= parameters.iterations {
                break
            }
        }
    }
}









func saveTrainingLossToFile(_ trainingLoss: Float) {
    let fileManager = FileManager.default
    if let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first {
        let flbDirectory = documentsDirectory.appendingPathComponent("Applications/FLB")

        do {
            if !fileManager.fileExists(atPath: flbDirectory.path) {
                try fileManager.createDirectory(at: flbDirectory, withIntermediateDirectories: true, attributes: nil)
                print("FLB directory created.")
            }

            let fileURL = flbDirectory.appendingPathComponent("training_loss.txt")
            let logEntry = "\(Date()): \(trainingLoss)\n"
            
//            print("wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww")
//            print(fileURL)

            if !fileManager.fileExists(atPath: fileURL.path) {
                try logEntry.write(to: fileURL, atomically: true, encoding: .utf8)
            } else {
                let fileHandle = try FileHandle(forUpdating: fileURL)
                fileHandle.seekToEndOfFile()
                if let data = logEntry.data(using: .utf8) {
                    fileHandle.write(data)
                }
                fileHandle.closeFile()
            }
        } catch {
            print("Error writing to file: \(error)")
        }
    } else {
        print("Documents directory not found.")
    }
}

 
 
func saveASCIIStringsToFile(_ asciiStrings: [String]) {
    let fileManager = FileManager.default
    if let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first {
        let flbDirectory = documentsDirectory.appendingPathComponent("Applications/FLB")

        do {
            // 检查并创建 FLB 目录
            if !fileManager.fileExists(atPath: flbDirectory.path) {
                try fileManager.createDirectory(at: flbDirectory, withIntermediateDirectories: true, attributes: nil)
                print("FLB directory created.")
            }

            let fileURL = flbDirectory.appendingPathComponent("ascii_data.txt")

            // 将数组内容拼接成一个字符串，每个元素占一行
            let logEntry = asciiStrings.joined(separator: "\n")

            // 检查文件是否存在
            if !fileManager.fileExists(atPath: fileURL.path) {
                // 文件不存在，创建并写入
                try logEntry.write(to: fileURL, atomically: true, encoding: .utf8)
            } else {
                // 文件存在，追加写入
                let fileHandle = try FileHandle(forUpdating: fileURL)
                fileHandle.seekToEndOfFile()
                if let data = ("\n" + logEntry).data(using: .utf8) {
                    fileHandle.write(data)
                }
                fileHandle.closeFile()
            }
        } catch {
            print("Error writing to file: \(error)")
        }
    } else {
        print("Documents directory not found.")
    }
}




func readASCIIStringsFromFile() -> [String]? {
    let fileManager = FileManager.default
    if let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first {
        let fileURL = documentsDirectory.appendingPathComponent("Applications/FLB/ascii_data.txt")

        do {
            // 读取文件内容
            let fileContents = try String(contentsOf: fileURL, encoding: .utf8)
            // 将内容按行分割成数组
            let asciiStrings = fileContents.components(separatedBy: "\n").filter { !$0.isEmpty }
            return asciiStrings
        } catch {
            print("Error reading from file: \(error)")
            return nil
        }
    } else {
        print("Documents directory not found.")
        return nil
    }
}




// ##################################################################################################################


func readTrainingLossFromFile() -> String? {
    let fileManager = FileManager.default
    if let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first {
        let flbDirectory = documentsDirectory.appendingPathComponent("Applications/FLB")
        let fileURL = flbDirectory.appendingPathComponent("training_loss.txt")
        
        // 检查文件是否存在
        if fileManager.fileExists(atPath: fileURL.path) {
            do {
                // 读取文件内容
                let content = try String(contentsOf: fileURL, encoding: .utf8)
                return content
            } catch {
                print("Error reading file: \(error)")
                return nil
            }
        } else {
            print("File does not exist at path: \(fileURL.path)")
            return nil
        }
    } else {
        print("Documents directory not found.")
        return nil
    }
}


import Foundation

func createAdapterURL(fileName: String) -> URL? {
    // 获取默认的文件管理器实例
    let fileManager = FileManager.default
    
    // 尝试获取用户的 Documents 目录 URL
    if let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first {
        
        // 定义适配器文件夹路径 "Applications/LoRAAdapters"
        let adapterDirectory = documentsDirectory.appendingPathComponent("Applications/LoRAAdapters")
        
        do {
            // 检查 LoRAAdapters 目录是否存在
            if !fileManager.fileExists(atPath: adapterDirectory.path) {
                // 如果目录不存在，创建 LoRAAdapters 目录及其中间目录
                try fileManager.createDirectory(at: adapterDirectory, withIntermediateDirectories: true, attributes: nil)
                print("LoRAAdapters directory created.")
            }
            
            // 在 LoRAAdapters 目录下生成完整的文件路径，并添加 `.safetensors` 扩展名
            let fileURL = adapterDirectory.appendingPathComponent("\(fileName).safetensors")
            return fileURL
        } catch {
            // 捕获文件夹创建过程中的错误，并打印错误信息
            print("Error creating directory: \(error)")
            return nil
        }
    } else {
        // 如果没有找到 Documents 目录，则打印错误信息
        print("Documents directory not found.")
        return nil
    }
}

 






import Foundation

func calculateMean(from data: String) -> Float? {
    // 正则表达式模式，用于匹配冒号后面的数值
    let pattern = #": ([\d.]+)"#
    let regex = try? NSRegularExpression(pattern: pattern, options: [])
    
    // 确保正则表达式创建成功
    guard let regex = regex else {
        print("Failed to create regex.")
        return nil
    }

    let nsrange = NSRange(data.startIndex..<data.endIndex, in: data)
    var values: [Float] = []

    // 匹配并提取数值
    regex.enumerateMatches(in: data, options: [], range: nsrange) { match, _, _ in
        if let match = match, let range = Range(match.range(at: 1), in: data) {
            let valueString = String(data[range])
            if let value = Float(valueString) {
                values.append(value)
            }
        }
    }

    // 确保有值可以计算均值
    guard !values.isEmpty else {
        print("No values found in the input data.")
        return nil
    }
    
    // 计算均值
    let mean = values.reduce(0, +) / Float(values.count)
    return mean
}





import Foundation

func getFileSize(at url: URL) -> Int64? {
    let fileManager = FileManager.default
    do {
        // 获取文件的属性
        let attributes = try fileManager.attributesOfItem(atPath: url.path)
        
        // 从属性中提取文件大小（字节数）
        if let fileSize = attributes[.size] as? Int64 {
            return fileSize
        }
    } catch {
        print("Error retrieving file size: \(error)")
    }
    return nil
}


// ##################################################################################################################



 
