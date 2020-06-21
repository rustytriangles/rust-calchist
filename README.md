# rust-calchist
I was struggling a bit trying to use [imgproc::calc_hist](https://docs.rs/opencv/0.19.2/opencv/imgproc/fn.calc_hist.html)
from opencv-rust. The documentation is a bit sparse. So once I figured it out, I created this simple example. This just
streams video capture, computes the histograms of the channels, and then overlays  them as line charts:

![screenshot](images/screenshot.png)

This assumes you're using OpenCV 4. There are a few differences in initialization for earlier versions.