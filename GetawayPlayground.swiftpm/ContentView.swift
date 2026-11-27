import SwiftUI

struct ContentView: View {
    @State private var hud = HUDModel()

    var body: some View {
        ZStack(alignment: .topLeading) {
            MetalTerrainView(hud: hud)
                .ignoresSafeArea()

            HUDOverlay(hud: hud)
                .padding()
                .allowsHitTesting(false)

            if let message = hud.errorMessage {
                Text(message)
                    .font(.headline)
                    .padding()
                    .background(.red.opacity(0.8), in: .rect(cornerRadius: 8))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }
}

private struct HUDOverlay: View {
    let hud: HUDModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(String(format: "%.0f fps   speed %.0f   altitude %.0f", hud.fps, hud.speed, hud.altitude))
            Text(String(format: "tessellation detail %.0f   max factor %d", hud.lodScale, hud.maxTessellation)
                 + (hud.wireframe ? "   wireframe" : "")
                 + (hud.lodFrozen ? "   LOD frozen" : ""))

            if hud.showHelp {
                Divider().frame(maxWidth: 360)
                ForEach(Self.helpLines, id: \.self) { Text($0) }
            }
        }
        .font(.system(.caption, design: .monospaced))
        .foregroundStyle(.white)
        .padding(10)
        .background(.black.opacity(0.45), in: .rect(cornerRadius: 8))
    }

    #if os(macOS)
    private static let helpLines = [
        "Click: capture mouse   Esc: release",
        "W A S D / arrows: move   E, Space: up   Q, C: down",
        "Shift: fast   Option: slow   Scroll, [ ]: speed",
        "F: wireframe   L: freeze LOD   + / -: detail",
        "R: reset camera   H: toggle help",
    ]
    #else
    private static let helpLines = [
        "Drag / mouse drag: look   Two fingers: fly forward",
        "Keyboard: W A S D move   E, Space: up   Q, C: down",
        "Shift: fast   Option: slow   Scroll, [ ]: speed",
        "F: wireframe   L: freeze LOD   + / -: detail",
        "R: reset camera   H: toggle help",
    ]
    #endif
}

