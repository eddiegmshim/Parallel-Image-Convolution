// Package png allows for loading png images and applying
// image flitering effects on them.
package png

import (
	"image"
	"image/color"
)

// Grayscale applies a grayscale filtering effect to the image
func (img *Image) Grayscale() {
	bounds := img.out.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, a := img.in.At(x, y).RGBA()
			greyC := clamp(float64(r+g+b) / 3) //perform grey out by averaging their rgb values
			img.out.Set(x, y, color.RGBA64{greyC, greyC, greyC, uint16(a)})
		}
	}
}

// Performs a sharpen effect
func (img *Image) Sharpen() {
	kernel := [3][3]float64{
		{0, -1, 0},
		{-1, 5, -1},
		{0, -1, 0},
	}

	bounds := img.out.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			v := img.kernelApply(x, y, kernel, bounds)
			img.out.Set(x, y, color.RGBA64{v[0], v[1], v[2], v[3]})
		}
	}
}

//Performs a edge-detection effect
func (img *Image) EdgeDetect(){
	kernel := [3][3]float64{
		{-1, -1, -1},
		{-1, 8, -1},
		{-1, -1, -1},
	}

	bounds := img.out.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			v := img.kernelApply(x, y, kernel, bounds)
			img.out.Set(x, y, color.RGBA64{v[0], v[1], v[2], v[3]})
		}
	}
}

//Performs a blur effect
func (img *Image) Blur(){
	kernel := [3][3]float64{
		{1.0/9.0, 1.0/9.0, 1.0/9.0},
		{1.0/9.0, 1.0/9.0, 1.0/9.0},
		{1.0/9.0, 1.0/9.0, 1.0/9.0},
	}

	bounds := img.out.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			v := img.kernelApply(x, y, kernel, bounds)
			img.out.Set(x, y, color.RGBA64{v[0], v[1], v[2], v[3]})
		}
	}
}

func(img * Image) kernelApply(x int, y int, kernel [3][3]float64, bounds image.Rectangle) [4]uint16 {
	rTransformed := float64(0)
	gTransformed := float64(0)
	bTransformed := float64(0)
	a := float64(0)

	for kRow := 0; kRow < 3; kRow++{
		for kCol := 0; kCol < 3; kCol++{
			imgRow := x + kRow - 1
			imgCol := y + kCol - 1
			if imgRow < 0 || imgRow > bounds.Max.X || imgCol < 0 || imgCol > bounds.Max.Y {
				//if index is out of bounds, pad with 0 values
			} else {
				r, g, b, a_temp := img.in.At(imgRow, imgCol).RGBA()

				// as defined by http://www.songho.ca/dsp/convolution/convolution2d_example.html
				// we need to flip kernel horizonal and vertical ways
				rTransformed += kernel[2-kRow][2-kCol] * float64(r)
				gTransformed += kernel[2-kRow][2-kCol] * float64(g)
				bTransformed += kernel[2-kRow][2-kCol] * float64(b)

				//take the original alpha value at our center coordinate
				if imgRow == x && imgCol == y {
					a = float64(a_temp)
				}
			}
		}
	}
	return [4]uint16{clamp(rTransformed), clamp(gTransformed), clamp(bTransformed), clamp(a)}
}
