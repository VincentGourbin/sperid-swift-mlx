// ScheduleAndSamplerTests.swift - Euler grid and systematic resampling
// Copyright 2026 Vincent Gourbin

import Testing
@testable import IdolCore

@Suite("FlowSchedule")
struct FlowScheduleTests {
    // numpy: t = linspace(1, 0, steps + 1); t = r t / (1 + (r - 1) t)
    @Test func turboGridMatchesNumpy() {
        let s = FlowSchedule(steps: 5, rescaleT: 3)
        let expected = [1.0, 0.9230769230769231, 0.818181818181818, 0.6666666666666666, 0.4285714285714285, 0.0]
        for (a, b) in zip(s.timesteps, expected) { #expect(abs(a - b) < 1e-12) }
        #expect(s.networkTime(0) == 1000)
    }

    @Test func defaultGridMatchesNumpy() {
        let s = FlowSchedule(steps: 50, rescaleT: 3)
        #expect(s.timesteps.count == 51)
        #expect(abs(s.timesteps[1] - 0.9932432432432432) < 1e-12)
        #expect(abs(s.timesteps[5] - 0.9642857142857144) < 1e-12)
        #expect(s.timesteps.last == 0)
    }

    @Test func turboPhaseSwitchesAtOneHalf() {
        let s = FlowSchedule(steps: 5, rescaleT: 3)
        #expect((0..<5).map { s.turboPhase($0) } == [.slow, .slow, .slow, .slow, .fast])
    }
}

@Suite("SystematicSampler")
struct SystematicSamplerTests {
    @Test func preservesTheCount() {
        for n in [1, 7, 32, 1000, 262_144] {
            let out = SystematicSampler.split(probabilities: [0.1, 0.2, 0, 0.3, 0.05, 0.05, 0.2, 0.1],
                                              count: n, offset: 0.37)
            #expect(out.reduce(0, +) == n)
            #expect(out[2] == 0)  // a zero-probability child never receives a point
        }
    }

    @Test func isStratified() {
        // 1000 points over [0.5, 0.25, 0.25]: each bin gets its share ± 1
        let out = SystematicSampler.split(probabilities: [0.5, 0.25, 0.25], count: 1000, offset: 0.5)
        #expect(abs(out[0] - 500) <= 1 && abs(out[1] - 250) <= 1 && abs(out[2] - 250) <= 1)
    }

    @Test func zeroRowIsUniform() {
        // numpy: searchsorted([.25, .5, .75, 1], (0.5 + arange(8)) / 8) → bincount [2, 2, 2, 2]
        #expect(SystematicSampler.split(probabilities: [0, 0, 0, 0], count: 8, offset: 0.5) == [2, 2, 2, 2])
        // offset 0 puts u = 0.25 on a boundary: side='left' sends it to bin 0 → [3, 2, 2, 1]
        #expect(SystematicSampler.split(probabilities: [0, 0, 0, 0], count: 8, offset: 0) == [3, 2, 2, 1])
    }
}
