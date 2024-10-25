const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build the shared library for `levvy`
    const zig_mod = b.addSharedLibrary(.{
        .name = "levvy",
        .root_source_file = b.path("src/levvy.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Define relative path for installation
    const copy_step = b.addInstallFile(zig_mod.getEmittedBin(), "lib/levvy.dll");

    // Make the built-in install step depend on the copy step
    b.getInstallStep().dependOn(&copy_step.step);

    // Test configuration for `levvy.zig`
    const lib_tests = b.addTest(.{
        .root_source_file = b.path("src/levvy.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);

    // Test configuration for `main.zig`
    const main_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_main_tests = b.addRunArtifact(main_tests);

    // Combine both test artifacts in the `test` step
    const test_step = b.step("test", "Run unit tests for levvy and main");
    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&run_main_tests.step);

    // Build main executable (for development/testing purposes)
    const exe = b.addExecutable(.{
        .name = "levvy_app",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_cmd = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run the main application");
    run_step.dependOn(&run_cmd.step);

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
}
