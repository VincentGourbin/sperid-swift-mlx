// SplatView.swift - RealityKit viewer of the generated splat (orbit with the mouse)
// Copyright 2026 Vincent Gourbin

import IdolCore
import RealityKit
import SwiftUI

/// A SwiftUI view showing a splat with orbit controls. Change `id` to swap the object.
@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
public struct SplatView: View {
    let cloud: GaussianCloud?
    let version: Int

    public init(cloud: GaussianCloud?, id version: Int) {
        self.cloud = cloud
        self.version = version
    }

    public var body: some View {
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
        #if os(macOS) || os(iOS)
        .realityViewCameraControls(.orbit)
        #endif
        .background(Color(white: 0.05))
    }
}

struct VersionComponent: Component {
    let version: Int
}
