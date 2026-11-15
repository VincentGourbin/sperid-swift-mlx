// SplatView.swift - RealityKit viewer of the generated splat (orbit with the mouse)
// Copyright 2026 Vincent Gourbin

import IdolCore
import RealityKit
import SwiftUI

@available(macOS 27.0, *)
struct SplatView: View {
    let cloud: GaussianCloud?
    let version: Int

    var body: some View {
        RealityView { content in
            let anchor = Entity()
            anchor.name = "splat-root"
            content.add(anchor)
        } update: { content in
            guard let root = content.entities.first(where: { $0.name == "splat-root" }) else { return }
            if root.components[VersionComponent.self]?.version == version { return }
            root.children.removeAll()
            if let cloud, let entity = try? SplatEntity.make(cloud) {
                root.addChild(entity)
            }
            root.components.set(VersionComponent(version: version))
        }
        .realityViewCameraControls(.orbit)
        .background(Color(white: 0.05))
    }
}

struct VersionComponent: Component {
    let version: Int
}
