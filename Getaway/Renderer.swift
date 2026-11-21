import MetalKit
import QuartzCore
import simd

enum TerrainConfig {
    static let patchesPerSide = 128
    static let patchSize: Float = 64
    static var terrainSize: Float { Float(patchesPerSide) * patchSize }
    static let heightScale: Float = 300
    static let baseFrequency: Float = 1 / 900
    static let waterLevel: Float = -45
    static let materialTextureSize = 1024
    static let defaultLODScale: Float = 64
    #if os(macOS)
    static let maxTessellationFactor = 64
    #else
    static let maxTessellationFactor = 16
    #endif
}

/// Must match `Uniforms` in ShaderCommon.h.
struct Uniforms {
    var viewProjection: simd_float4x4
    var invViewProjection: simd_float4x4
    var cameraPosition: SIMD4<Float>
    var lodOrigin: SIMD4<Float>
    var sunDirection: SIMD4<Float>
    var frustumPlanes: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
    var terrainParams: SIMD4<Float>
    var tessParams: SIMD4<Float>
}

enum RendererError: Error, CustomStringConvertible {
    case noDevice
    case missingFunction(String)
    case resourceCreation(String)

    var description: String {
        switch self {
        case .noDevice: "Metal is not available on this device."
        case .missingFunction(let name): "Missing Metal function \(name)."
        case .resourceCreation(let what): "Could not create \(what)."
        }
    }
}

final class Renderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    private let tessFactorPipeline: MTLComputePipelineState
    private let terrainPipeline: MTLRenderPipelineState
    private let terrainWirePipeline: MTLRenderPipelineState
    private let skyPipeline: MTLRenderPipelineState
    private let waterPipeline: MTLRenderPipelineState
    private let cloudPipeline: MTLRenderPipelineState

    private let terrainDepthState: MTLDepthStencilState
    private let wireDepthState: MTLDepthStencilState
    private let skyDepthState: MTLDepthStencilState
    private let waterDepthState: MTLDepthStencilState
    private var sceneColorTexture: MTLTexture?
    private var sceneDepthTexture: MTLTexture?
    private var sceneTextureSize = MTLSize(width: 0, height: 0, depth: 1)

    private let controlPointBuffer: MTLBuffer
    private let tessFactorBuffer: MTLBuffer
    private let patchCount: Int
    private let materialTexture: MTLTexture
    private let materialSampler: MTLSamplerState

    private let input: InputState
    private let hud: HUDModel
    private var camera = FlyCamera()
    private var lodScale = TerrainConfig.defaultLODScale
    private var wireframe = false
    private var frozenLODOrigin: SIMD3<Float>?
    private let sunDirection = normalize(SIMD3<Float>(0.62, 0.34, -0.42))

    private let startTime = CACurrentMediaTime()
    private var lastFrameTime = CACurrentMediaTime()
    private var fpsAccumulator: (frames: Int, time: Double) = (0, 0)
    init(view: MTKView, input: InputState, hud: HUDModel) throws {
        guard let device = view.device else { throw RendererError.noDevice }
        guard let queue = device.makeCommandQueue() else { throw RendererError.resourceCreation("command queue") }
        guard let library = device.makeDefaultLibrary() else { throw RendererError.resourceCreation("shader library") }

        self.device = device
        self.commandQueue = queue
        self.input = input
        self.hud = hud

        func function(_ name: String) throws -> MTLFunction {
            guard let fn = library.makeFunction(name: name) else { throw RendererError.missingFunction(name) }
            return fn
        }

        tessFactorPipeline = try device.makeComputePipelineState(function: try function("terrainTessFactors"))

        let terrainDescriptor = MTLRenderPipelineDescriptor()
        terrainDescriptor.label = "Terrain"
        terrainDescriptor.vertexFunction = try function("terrainVertex")
        terrainDescriptor.fragmentFunction = try function("terrainFragment")
        terrainDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        terrainDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat

        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float2
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<SIMD2<Float>>.stride
        vertexDescriptor.layouts[0].stepFunction = .perPatchControlPoint
        vertexDescriptor.layouts[0].stepRate = 1
        terrainDescriptor.vertexDescriptor = vertexDescriptor

        terrainDescriptor.tessellationFactorStepFunction = .perPatch
        terrainDescriptor.tessellationFactorFormat = .half
        terrainDescriptor.tessellationPartitionMode = .fractionalEven
        terrainDescriptor.tessellationOutputWindingOrder = .counterClockwise
        terrainDescriptor.tessellationControlPointIndexType = .none
        terrainDescriptor.maxTessellationFactor = TerrainConfig.maxTessellationFactor
        terrainDescriptor.isTessellationFactorScaleEnabled = false

        terrainPipeline = try device.makeRenderPipelineState(descriptor: terrainDescriptor)

        terrainDescriptor.label = "Terrain Wireframe"
        terrainDescriptor.fragmentFunction = try function("terrainWireFragment")
        terrainWirePipeline = try device.makeRenderPipelineState(descriptor: terrainDescriptor)

        let skyDescriptor = MTLRenderPipelineDescriptor()
        skyDescriptor.label = "Sky"
        skyDescriptor.vertexFunction = try function("skyVertex")
        skyDescriptor.fragmentFunction = try function("skyFragment")
        skyDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        skyDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat
        skyPipeline = try device.makeRenderPipelineState(descriptor: skyDescriptor)

        let waterDescriptor = MTLRenderPipelineDescriptor()
        waterDescriptor.label = "Water"
        waterDescriptor.vertexFunction = try function("waterVertex")
        waterDescriptor.fragmentFunction = try function("waterFragment")
        waterDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        waterDescriptor.colorAttachments[0].isBlendingEnabled = true
        waterDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        waterDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        waterDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        waterDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        waterDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat
        waterPipeline = try device.makeRenderPipelineState(descriptor: waterDescriptor)

        let cloudDescriptor = MTLRenderPipelineDescriptor()
        cloudDescriptor.label = "Volumetric Clouds"
        cloudDescriptor.vertexFunction = try function("skyVertex")
        cloudDescriptor.fragmentFunction = try function("cloudFragment")
        cloudDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        cloudPipeline = try device.makeRenderPipelineState(descriptor: cloudDescriptor)

        func depthState(_ compare: MTLCompareFunction, write: Bool) throws -> MTLDepthStencilState {
            let descriptor = MTLDepthStencilDescriptor()
            descriptor.depthCompareFunction = compare
            descriptor.isDepthWriteEnabled = write
            guard let state = device.makeDepthStencilState(descriptor: descriptor) else {
                throw RendererError.resourceCreation("depth state")
            }
            return state
        }
        // Reversed Z: larger depth values are closer.
        terrainDepthState = try depthState(.greater, write: true)
        wireDepthState = try depthState(.greaterEqual, write: false)
        skyDepthState = try depthState(.always, write: false)
        waterDepthState = try depthState(.greater, write: true)

        let n = TerrainConfig.patchesPerSide
        let size = TerrainConfig.patchSize
        let origin = -TerrainConfig.terrainSize * 0.5
        var controlPoints: [SIMD2<Float>] = []
        controlPoints.reserveCapacity(n * n * 4)
        for z in 0..<n {
            for x in 0..<n {
                let x0 = origin + Float(x) * size, x1 = origin + Float(x + 1) * size
                let z0 = origin + Float(z) * size, z1 = origin + Float(z + 1) * size
                controlPoints += [SIMD2(x0, z0), SIMD2(x1, z0), SIMD2(x1, z1), SIMD2(x0, z1)]
            }
        }
        patchCount = n * n
        guard let cpBuffer = device.makeBuffer(bytes: controlPoints,
                                               length: MemoryLayout<SIMD2<Float>>.stride * controlPoints.count,
                                               options: .storageModeShared),
              let tfBuffer = device.makeBuffer(length: MemoryLayout<MTLQuadTessellationFactorsHalf>.stride * patchCount,
                                               options: .storageModePrivate)
        else { throw RendererError.resourceCreation("terrain buffers") }
        cpBuffer.label = "Terrain Control Points"
        tfBuffer.label = "Terrain Tessellation Factors"
        controlPointBuffer = cpBuffer
        tessFactorBuffer = tfBuffer

        materialTexture = try Renderer.makeMaterialTexture(device: device, queue: queue, library: library)

        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        samplerDescriptor.mipFilter = .linear
        samplerDescriptor.sAddressMode = .repeat
        samplerDescriptor.tAddressMode = .repeat
        samplerDescriptor.maxAnisotropy = 8
        guard let sampler = device.makeSamplerState(descriptor: samplerDescriptor) else {
            throw RendererError.resourceCreation("sampler")
        }
        materialSampler = sampler

        super.init()
        hud.maxTessellation = TerrainConfig.maxTessellationFactor
    }

    private static func makeMaterialTexture(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) throws -> MTLTexture {
        let size = TerrainConfig.materialTextureSize
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type2DArray
        descriptor.pixelFormat = .rgba8Unorm
        descriptor.width = size
        descriptor.height = size
        descriptor.arrayLength = 4
        descriptor.mipmapLevelCount = Int(log2(Double(size))) + 1
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private

        guard let texture = device.makeTexture(descriptor: descriptor),
              let generator = library.makeFunction(name: "generateMaterials"),
              let commandBuffer = queue.makeCommandBuffer(),
              let compute = commandBuffer.makeComputeCommandEncoder()
        else { throw RendererError.resourceCreation("material textures") }
        texture.label = "Ground Materials"

        let pipeline = try device.makeComputePipelineState(function: generator)
        compute.setComputePipelineState(pipeline)
        compute.setTexture(texture, index: 0)
        let threads = MTLSize(width: 8, height: 8, depth: 1)
        let groups = MTLSize(width: (size + 7) / 8, height: (size + 7) / 8, depth: 4)
        compute.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        compute.endEncoding()

        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw RendererError.resourceCreation("mipmap encoder")
        }
        blit.generateMipmaps(for: texture)
        blit.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return texture
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        sceneColorTexture = nil
        sceneDepthTexture = nil
    }

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        let dt = Float(min(now - lastFrameTime, 0.1))
        lastFrameTime = now

        handleActions()
        camera.update(deltaTime: dt, input: input)

        var uniforms = makeUniforms(drawableSize: view.drawableSize, time: Float(now - startTime))
        var patchCount32 = UInt32(patchCount)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        commandBuffer.label = "Frame"

        if let compute = commandBuffer.makeComputeCommandEncoder() {
            compute.label = "Tessellation Factors"
            compute.setComputePipelineState(tessFactorPipeline)
            compute.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 0)
            compute.setBuffer(controlPointBuffer, offset: 0, index: 1)
            compute.setBuffer(tessFactorBuffer, offset: 0, index: 2)
            compute.setBytes(&patchCount32, length: MemoryLayout<UInt32>.stride, index: 3)
            let width = tessFactorPipeline.threadExecutionWidth
            compute.dispatchThreadgroups(MTLSize(width: (patchCount + width - 1) / width, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            compute.endEncoding()
        }

        guard let drawable = view.currentDrawable,
              ensureSceneTextures(size: view.drawableSize, colorFormat: view.colorPixelFormat),
              let sceneColorTexture,
              let sceneDepthTexture
        else {
            commandBuffer.commit()
            return
        }

        let scenePass = MTLRenderPassDescriptor()
        scenePass.colorAttachments[0].texture = sceneColorTexture
        scenePass.colorAttachments[0].loadAction = .clear
        scenePass.colorAttachments[0].storeAction = .store
        scenePass.colorAttachments[0].clearColor = view.clearColor
        scenePass.depthAttachment.texture = sceneDepthTexture
        scenePass.depthAttachment.loadAction = .clear
        scenePass.depthAttachment.storeAction = .store
        scenePass.depthAttachment.clearDepth = 0

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: scenePass) else {
            commandBuffer.commit()
            return
        }
        encoder.label = "Scene"

        encoder.setRenderPipelineState(skyPipeline)
        encoder.setDepthStencilState(skyDepthState)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        encoder.setCullMode(.none)
        encoder.setTessellationFactorBuffer(tessFactorBuffer, offset: 0, instanceStride: 0)
        encoder.setVertexBuffer(controlPointBuffer, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)

        encoder.setRenderPipelineState(terrainPipeline)
        encoder.setDepthStencilState(terrainDepthState)
        encoder.setFragmentTexture(materialTexture, index: 0)
        encoder.setFragmentSamplerState(materialSampler, index: 0)
        drawTerrainPatches(encoder)

        if wireframe {
            encoder.setRenderPipelineState(terrainWirePipeline)
            encoder.setDepthStencilState(wireDepthState)
            encoder.setTriangleFillMode(.lines)
            drawTerrainPatches(encoder)
            encoder.setTriangleFillMode(.fill)
        }

        encoder.setRenderPipelineState(waterPipeline)
        encoder.setDepthStencilState(waterDepthState)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)

        encoder.endEncoding()

        let compositePass = MTLRenderPassDescriptor()
        compositePass.colorAttachments[0].texture = drawable.texture
        compositePass.colorAttachments[0].loadAction = .dontCare
        compositePass.colorAttachments[0].storeAction = .store

        if let composite = commandBuffer.makeRenderCommandEncoder(descriptor: compositePass) {
            composite.label = "Cloud Composite"
            composite.setRenderPipelineState(cloudPipeline)
            composite.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 0)
            composite.setFragmentTexture(sceneColorTexture, index: 0)
            composite.setFragmentTexture(sceneDepthTexture, index: 1)
            composite.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            composite.endEncoding()
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()

        updateHUD(frameTime: Double(dt))
    }

    private func drawTerrainPatches(_ encoder: MTLRenderCommandEncoder) {
        encoder.drawPatches(numberOfPatchControlPoints: 4,
                            patchStart: 0,
                            patchCount: patchCount,
                            patchIndexBuffer: nil,
                            patchIndexBufferOffset: 0,
                            instanceCount: 1,
                            baseInstance: 0)
    }

    private func ensureSceneTextures(size: CGSize, colorFormat: MTLPixelFormat) -> Bool {
        let width = max(Int(size.width), 1)
        let height = max(Int(size.height), 1)
        guard sceneColorTexture == nil || sceneDepthTexture == nil ||
              sceneTextureSize.width != width || sceneTextureSize.height != height else {
            return true
        }

        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: colorFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        colorDescriptor.usage = [.renderTarget, .shaderRead]
        colorDescriptor.storageMode = .private

        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        depthDescriptor.usage = [.renderTarget, .shaderRead]
        depthDescriptor.storageMode = .private

        sceneColorTexture = device.makeTexture(descriptor: colorDescriptor)
        sceneDepthTexture = device.makeTexture(descriptor: depthDescriptor)
        sceneColorTexture?.label = "Scene Color"
        sceneDepthTexture?.label = "Scene Depth"
        sceneTextureSize = MTLSize(width: width, height: height, depth: 1)
        return sceneColorTexture != nil && sceneDepthTexture != nil
    }

    private func makeUniforms(drawableSize: CGSize, time: Float) -> Uniforms {
        let aspect = Float(drawableSize.width / max(drawableSize.height, 1))
        let projection = simd_float4x4.perspectiveReversedZ(fovY: .pi / 3, aspect: max(aspect, 0.01),
                                                            near: 0.5, far: 30_000)
        let viewProjection = projection * camera.viewMatrix
        let planes = viewProjection.frustumSidePlanes
        let lodOrigin = frozenLODOrigin ?? camera.position

        return Uniforms(
            viewProjection: viewProjection,
            invViewProjection: viewProjection.inverse,
            cameraPosition: SIMD4(camera.position, time),
            lodOrigin: SIMD4(lodOrigin, 0),
            sunDirection: SIMD4(sunDirection, 0),
            frustumPlanes: (planes[0], planes[1], planes[2], planes[3]),
            terrainParams: SIMD4(TerrainConfig.heightScale, TerrainConfig.baseFrequency,
                                 TerrainConfig.waterLevel, TerrainConfig.terrainSize),
            tessParams: SIMD4(Float(TerrainConfig.maxTessellationFactor), lodScale, TerrainConfig.patchSize, 0)
        )
    }

    private func handleActions() {
        for action in input.consumeActions() {
            switch action {
            case .toggleWireframe: wireframe.toggle()
            case .toggleHelp: hud.showHelp.toggle()
            case .toggleLODFreeze: frozenLODOrigin = frozenLODOrigin == nil ? camera.position : nil
            case .increaseDetail: lodScale = min(lodScale * 1.25, 512)
            case .decreaseDetail: lodScale = max(lodScale / 1.25, 4)
            case .faster: camera.speed = min(camera.speed * 1.5, 4000)
            case .slower: camera.speed = max(camera.speed / 1.5, 2)
            case .resetCamera: camera.reset()
            }
        }
    }

    private func updateHUD(frameTime: Double) {
        fpsAccumulator.frames += 1
        fpsAccumulator.time += frameTime
        guard fpsAccumulator.time >= 0.25 else { return }

        hud.fps = Double(fpsAccumulator.frames) / fpsAccumulator.time
        fpsAccumulator = (0, 0)
        hud.speed = camera.speed
        hud.altitude = camera.position.y
        hud.lodScale = lodScale
        hud.wireframe = wireframe
        hud.lodFrozen = frozenLODOrigin != nil
        hud.mouseCaptured = input.mouseCaptured
    }
}
