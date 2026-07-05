const std = @import("std");

pub fn main() !void {
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});
}

test "simple test" {
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(std.testing.allocator); // try commenting this out and see if zig detects the memory leak!
    try list.append(std.testing.allocator, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop().?);
}
