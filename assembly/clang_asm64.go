package assembly

import "unsafe"

//go:linkname clang_one clang_one
func clang_one(fn unsafe.Pointer, x float64) float64

//go:linkname clang_two clang_two
func clang_two(fn unsafe.Pointer, x, y float64) float64

//go:linkname clang_three clang_three
func clang_three(fn unsafe.Pointer, x float64, y float64, z float64) float64

func ClangOne(fn unsafe.Pointer, x float64) float64 {
	return clang_one(fn, x)
}

func ClangTwo(fn unsafe.Pointer, x, y float64) float64 {
	return clang_two(fn, x, y)
}

func ClangThree(fn unsafe.Pointer, x, y, z float64) float64 {
	return clang_three(fn, x, y, z)
}
