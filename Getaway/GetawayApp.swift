//
//  GetawayApp.swift
//  Getaway
//
//  Created by Jocelyn Manger on 11/3/26.
//

import SwiftUI

@main
struct GetawayApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        #if os(macOS)
        .defaultSize(width: 1440, height: 900)
        #endif
    }
}
