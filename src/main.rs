use std::{thread, time::Duration};

use opencv::{
    core,
    highgui,
    imgproc,
    prelude::*,
    types,
    videoio,
};

fn make_veci32(i: i32) -> types::VectorOfi32 {
    let mut result = types::VectorOfi32::with_capacity(1);
    result.push(i);
    return result;
}

fn make_vecf32(v0: f32, v1: f32) -> types::VectorOff32 {
    let mut result = types::VectorOff32::with_capacity(2);
    result.push(v0);
    result.push(v1);
    return result;
}

fn compute_pt(x: i32, val: f32, num_buckets: i32, max_count: f32,
	      image_width: i32, image_height: i32,
	      bounds: &core::Rect) -> core::Point {
    let tx = x as f32 / num_buckets as f32;
    let ty = 1. - val / max_count;
    let ix = bounds.x + (tx * bounds.width as f32) as i32;
    let iy = bounds.y + (ty * bounds.height as f32) as i32;
    return core::Point { x: ix, y: iy};
}

fn compute_histogram(images: &types::VectorOfMat, channel: i32, num_buckets: i32) -> Result<Mat, opencv::Error> {

    let mut hist = Mat::default()?;
    let channels = make_veci32(channel);

    let mask = Mat::default()?;
    let hist_size = make_veci32(num_buckets);
    let ranges = make_vecf32(0., 255.);
    let accumulate = false;

    match imgproc::calc_hist(&images, &channels, &mask,
                             &mut hist,
                             &hist_size, &ranges, accumulate) {
        Result::Ok(_) => {
            // println!("hist = {:?}", r_hist);
        },
        Result::Err(err) => println!("error in calc_hist {:?}", err)
    }

    return Ok(hist);
}

fn draw_histogram(mut frame: &mut Mat, r_hist: Mat, g_hist: Mat, b_hist: Mat, bounds: core::Rect) {

    let hist_size = r_hist.rows();

    let mut frame_width = 0;
    let mut frame_height = 0;
    match frame.size() {
        Result::Ok(s) => {
            frame_width = s.width;
            frame_height = s.height;
        }
        Err(_) => { }
    }

    let red   = core::Scalar::new(  0.,  0.,255., 0.);
    let green = core::Scalar::new(  0.,255.,  0., 0.);
    let blue  = core::Scalar::new(255.,  0.,  0., 0.);
    let black = core::Scalar::new(  0.,  0.,  0., 0.);
    let _white = core::Scalar::new(255.,255.,255., 0.);
    let line_width = 1;
    let _fill_type = imgproc::LineTypes::FILLED as i32;
    let line_type = imgproc::LineTypes::LINE_8 as i32;
    let shift = 0;

    let mut max_count = 0. as f32;
    for i in 0..r_hist.rows() {
        match r_hist.at(i) {
            Result::Ok(v) => {
                max_count = max_count.max(*v);
            }
            Err(_) => { }
        }
        match g_hist.at(i) {
            Result::Ok(v) => {
                max_count = max_count.max(*v);
            }
            Err(_) => { }
        }
        match b_hist.at(i) {
            Result::Ok(v) => {
                max_count = max_count.max(*v);
            }
            Err(_) => { }
        }
    }

    let ll = compute_pt(0,0.,hist_size, max_count,frame_width, frame_height, &bounds);
    let ur = compute_pt(hist_size,max_count, hist_size,max_count,frame_width,frame_height, &bounds);
    let outline = core::Rect { x: ll.x, y: ur.y, width: ur.x-ll.x, height: ll.y-ur.y };
// background
//    let r = imgproc::rectangle(&mut frame, outline,
//                               _white, _fill_type, line_type, shift);

    let mut r_prev = 0 as f32;
    match r_hist.at(0) {
        Result::Ok(prev) => {
            r_prev = *prev;
        }
        Err(_) => { }
    }

    let mut g_prev = 0 as f32;
    match g_hist.at(0) {
        Result::Ok(prev) => {
            g_prev = *prev;
        }
        Err(_) => { }
    }

    let mut b_prev = 0 as f32;
    match b_hist.at(0) {
        Result::Ok(prev) => {
            b_prev = *prev;
        }
        Err(_) => { }
    }

    for i in 1..hist_size {
        match r_hist.at(i) {
            Result::Ok(next) => {
                let r_next = *next;
                let _ = imgproc::line(&mut frame,
                                      compute_pt(i-1, r_prev, hist_size, max_count, frame_width, frame_height, &bounds),
                                      compute_pt(i, r_next, hist_size, max_count, frame_width, frame_height, &bounds),
                                      red,
                                      line_width, line_type, shift);
                r_prev = r_next;
            }
            Err(_) => { }
        }

        match g_hist.at(i) {
            Result::Ok(next) => {
                let g_next = *next;
                let _ = imgproc::line(&mut frame,
                                      compute_pt(i-1, g_prev, hist_size, max_count, frame_width, frame_height, &bounds),
                                      compute_pt(i, g_next, hist_size, max_count, frame_width, frame_height, &bounds),
                                      green,
                                      line_width, line_type, shift);
                g_prev = g_next;
            }
            Err(_) => { }
        }

        match b_hist.at(i) {
            Result::Ok(next) => {
                let b_next = *next;
                let _ = imgproc::line(&mut frame,
                                      compute_pt(i-1, b_prev, hist_size, max_count, frame_width, frame_height, &bounds),
                                      compute_pt(i, b_next, hist_size, max_count, frame_width, frame_height, &bounds),
                                      blue,
                                      line_width, line_type, shift);
                b_prev = b_next;
            }
            Err(_) => { }
        }
    }


    let _ = imgproc::rectangle(&mut frame, outline,
                               black, line_width, line_type,shift);
}

fn run() -> opencv::Result<()> {

    let num_buckets = 256;
    let hist_width = 600;
    let hist_height = 150;
    let hist_pad = 32;

    // Create a window
    let window = "video capture";
    highgui::named_window(window, 1)?;

    // Open the video camera
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;  // 0 is the default camera
    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Unable to open default camera!");
    }

    loop {
        let mut frame = Mat::default()?;

        cam.read(&mut frame)?;
        if frame.size()?.width == 0 {
            thread::sleep(Duration::from_secs(50));
            continue;
        }

        // compute histograms
        let mut images = types::VectorOfMat::with_capacity(1);
        images.push(frame.clone()?); // wassup with clone?

        let r_hist = compute_histogram(&images, 2, num_buckets)?;
        let g_hist = compute_histogram(&images, 1, num_buckets)?;
        let b_hist = compute_histogram(&images, 0, num_buckets)?;

        // draw the histogram
	let hist_bounds = core::Rect {
	    x: (frame.size()?.width - hist_width) / 2,
	    y: frame.size()?.height - (hist_height + hist_pad),
	    width: hist_width,
	    height: hist_height
            };
        draw_histogram(&mut frame, r_hist, g_hist, b_hist, hist_bounds);

        // Display the image
        highgui::imshow(window, &frame)?;

        // Wait 10 milliseconds and quit if a key was pressed
        if highgui::wait_key(10)? > 0 {
            break;
        }
    }
    Ok(())
}

fn main() {
    run().unwrap()
}
