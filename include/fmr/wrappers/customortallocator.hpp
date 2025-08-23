#pragma once

#include <onnxruntime_cxx_api.h>

namespace fmr {

template<class... Ts>
struct Overload : Ts... { using Ts::operator()...; };

template<class... Ts>
Overload(Ts...) -> Overload<Ts...>;

// Just a class to simplify the use of Ort::Allocator and Ort::AllocatorWithDefaultOptions
class custom_ort_alloc {
public:
    using AllocatorVariant = std::variant<Ort::AllocatorWithDefaultOptions
                                          , Ort::Allocator>;

    custom_ort_alloc()
        : m_allocator(std::move(Ort::AllocatorWithDefaultOptions())) {}

    custom_ort_alloc(Ort::AllocatorWithDefaultOptions default_alloc)
        : m_allocator(std::move(default_alloc)) {}

    custom_ort_alloc(Ort::Allocator specific_alloc)
        : m_allocator(std::move(specific_alloc)) {}

    // Returns the underlying OrtAllocator*
    OrtAllocator* get() const {
        return std::visit(Overload {
                              [](const Ort::AllocatorWithDefaultOptions& alloc) {
                                  return static_cast<OrtAllocator*>(alloc);
                              },
                              [](const Ort::Allocator& alloc) {
                                  return static_cast<OrtAllocator*>(alloc);
                              }
                          }, m_allocator);
    }

    OrtAllocator* operator->() const {
        return get();
    }

private:
    AllocatorVariant m_allocator;
};

}

