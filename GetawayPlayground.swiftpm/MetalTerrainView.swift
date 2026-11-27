import MetalKit
import SwiftUI

final class TerrainViewCoordinator {
    var renderer: Renderer?
}

@MainActor
private func makeTerrainView(hud: HUDModel, coordinator: TerrainViewCoordinator) -> TerrainMTKView {
    let view = TerrainMTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
    view.colorPixelFormat = .bgra8Unorm_srgb
    view.depthStencilPixelFormat = .depth32Float
    view.clearDepth = 0
    view.clearColor = MTLClearColor(red: 0.6, green: 0.7, blue: 0.85, alpha: 1)
    view.preferredFramesPerSecond = 120
    do {
        let renderer = try Renderer(view: view, input: view.input, hud: hud)
        coordinator.renderer = renderer
        view.delegate = renderer
    } catch {
        hud.errorMessage = "\(error)"
        view.isPaused = true
    }
    return view
}

#if os(macOS)
struct MetalTerrainView: NSViewRepresentable {
    let hud: HUDModel

    func makeCoordinator() -> TerrainViewCoordinator { TerrainViewCoordinator() }

    func makeNSView(context: Context) -> TerrainMTKView {
        makeTerrainView(hud: hud, coordinator: context.coordinator)
    }

    func updateNSView(_ nsView: TerrainMTKView, context: Context) {}
}
#else
struct MetalTerrainView: UIViewRepresentable {
    let hud: HUDModel

    func makeCoordinator() -> TerrainViewCoordinator { TerrainViewCoordinator() }

    func makeUIView(context: Context) -> TerrainMTKView {
        makeTerrainView(hud: hud, coordinator: context.coordinator)
    }

    func updateUIView(_ uiView: TerrainMTKView, context: Context) {}
}
#endif
