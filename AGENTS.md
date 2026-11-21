# Getaway Agent Guide

## Project

Getaway is a SwiftUI and Metal procedural-terrain demo targeting macOS, iOS/iPadOS, and visionOS. The renderer uses GPU tessellation, procedural material generation, reversed-Z depth, and a depth-aware volumetric atmosphere.

## Build and validation

- Open `Getaway.xcodeproj` and use the `Getaway` scheme.
- Command-line macOS build:

  ```sh
  xcodebuild -project Getaway.xcodeproj -scheme Getaway -destination 'platform=macOS' build
  ```

- Prefer Xcode diagnostics for focused Swift or Metal changes, then run a full build before handing off.
- Use the Swift Testing framework for unit tests and XCUIAutomation for UI tests.

## Architecture

- `Getaway/Renderer.swift` owns Metal resources, pipeline creation, per-frame encoding, and render-target resizing.
- `Getaway/Terrain.metal` contains terrain tessellation, surface shading, sky, water, volumetric clouds, and atmospheric lighting.
- `Getaway/ShaderCommon.h` contains the shared Metal uniform layout and common terrain/atmosphere functions.
- `Getaway/MaterialTextures.metal` generates the material texture array.
- `Getaway/TerrainMTKView.swift` translates platform input into `InputState`; keep rendering logic out of the view.
- `Getaway/ContentView.swift` contains only the SwiftUI host and HUD.

## Rendering invariants

- Keep Swift `Uniforms` in `Renderer.swift` byte-for-byte layout-compatible with Metal `Uniforms` in `ShaderCommon.h`.
- Depth is reversed-Z: clear depth is `0`, and closer geometry has a larger depth value.
- The scene renders to shader-readable color and depth textures before the cloud composite pass.
- Terrain patch edges must derive tessellation factors identically on both adjacent patches to prevent cracks.
- Recreate size-dependent render targets whenever `MTKView.drawableSize` changes.
- Respect platform tessellation limits: 64 on macOS and 16 elsewhere.

## Code conventions

- Use four-space indentation and existing Swift/Metal naming conventions.
- Prefer strong Swift types, `let` for constants, and async/await over Combine.
- Avoid force unwrapping.
- Keep changes scoped and do not alter signing settings or bundle identifiers unless requested.
- Update `README.md` when controls, requirements, major rendering features, or project layout change.
