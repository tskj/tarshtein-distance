const std = @import("std");

export fn fuzzy_search(query: [*]const u8, number_of_lines: c_uint, input: [*][*]const u8, output: [*]u16) callconv(.C) c_int {
    var q_len: usize = 0;
    while (query[q_len] != 0) : (q_len += 1) {}

    var longest_line_length: usize = 0;
    for (input, 0..number_of_lines) |line, index| {
        var l_len: u16 = 0;
        while (line[l_len] != 0) : (l_len += 1) {}
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

    buffer[0] = 10;
    buffer[1] = 20;
    buffer[2] = 30;

    output[0] = 100;
    output[1] = 101;
    output[2] = 102;

    return 0;
}
