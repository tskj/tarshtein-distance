const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const levvy_mod = b.createModule(.{
        .root_source_file = b.path("src/levvy.zig"),
        .target = target,
        .optimize = optimize,
    });

    const main_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Build the shared library for `levvy`
    const lib = b.addLibrary(.{
        .name = "levvy",
        .linkage = .dynamic,
        .root_module = levvy_mod,
    });

    // Install into zig-out/lib (levvy.dll on Windows, liblevvy.so on Linux)
    const install_lib = b.addInstallArtifact(lib, .{
        .dest_dir = .{ .override = .lib },
    });
    b.getInstallStep().dependOn(&install_lib.step);

    // Test configuration for `levvy.zig`
    const lib_tests = b.addTest(.{ .root_module = levvy_mod });
    const run_lib_tests = b.addRunArtifact(lib_tests);

    // Test configuration for `main.zig`
    const main_tests = b.addTest(.{ .root_module = main_mod });
    const run_main_tests = b.addRunArtifact(main_tests);

    // Combine both test artifacts in the `test` step
    const test_step = b.step("test", "Run unit tests for levvy and main");
    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&run_main_tests.step);

    // Build main executable (for development/testing purposes)
    const exe = b.addExecutable(.{
        .name = "levvy_app",
        .root_module = main_mod,
    });

    const run_cmd = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run the main application");
    run_step.dependOn(&run_cmd.step);

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
}
