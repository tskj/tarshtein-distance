const std = @import("std");

export fn fuzzy_search(query: [*:0]const u8, number_of_lines: c_uint, input: [*][*:0]const u8, output: [*]u16) callconv(.C) c_int {
    const q_len: u16 = @as(u16, @intCast(std.mem.len(query)));

    var longest_line_length: u16 = 0;
    for (input, 0..number_of_lines) |line, index| {
        const l_len = @as(u16, @intCast(std.mem.len(line)));
        output[index] = l_len;
        if (l_len > longest_line_length) longest_line_length = l_len;
    }

    // two rows are needed at a time
    // and two tables, one for streak and one for non-streak
    const memory_requirement = longest_line_length * 2 * 2;

    const allocator = std.heap.page_allocator;
    var buffer = allocator.alloc(u16, memory_requirement) catch {
        return -1;
    };
    defer allocator.free(buffer);

    // testing
    buffer[0] = 10;
    buffer[1] = 20;
    buffer[2] = 30;

    // testing
    output[0] = q_len;
    output[1] = longest_line_length;
    output[2] = @as(u16, @intCast(number_of_lines));

    return 0; // probably want to return best (lowest) distance
}
