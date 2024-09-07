fn main() {
    let height = 100;
    let width = 200;
    print!("P3\n{} {}\n255\n", width, height);
    for j in 0..height {
        for i in 0..width {
            let r: f64 = i as f64 / (width-1) as f64;
            let g: f64 = j as f64 / (height-1) as f64;
            let b: f64 = 0.0;
            print!("{} {} {}\n", (r * 255.0) as i32, (g * 255.0) as i32, (b * 255.0) as i32);
        }
    }
}
