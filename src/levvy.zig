const std = @import("std");

export fn fuzzy_search(query: [*]const u8, number_of_lines: c_int, input: [*][*]const u8, output: [*]u16) callconv(.C) c_int {
    _ = number_of_lines;
    _ = input;
    _ = query;

    output[0] = 1;
    output[1] = 2;
    output[2] = 3;

    return 0;
}
