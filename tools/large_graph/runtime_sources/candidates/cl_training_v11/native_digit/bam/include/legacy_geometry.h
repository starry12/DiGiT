#pragma once
#include <cstdint>
#include <stdexcept>

// Logical cache lines may be smaller than the controller's DMA/PRP page.
inline uint64_t legacy_subrow_bytes(uint64_t line, uint64_t controller) {
    if (!line || !controller || (line & (line - 1)) ||
        (controller & (controller - 1)) || line < 512 ||
        (line > controller && (line % controller || line / controller > 64)))
        throw std::invalid_argument("Invalid legacy cache/PRP geometry");
    return line < controller ? line : controller;
}
inline uint64_t legacy_dma_offset(uint64_t slot, uint64_t line,
                                  uint64_t subrow, uint64_t controller) {
    uint64_t width = legacy_subrow_bytes(line, controller);
    if (subrow >= line / width)
        throw std::out_of_range("Invalid cache subrow");
    return slot * line + subrow * width;
}
