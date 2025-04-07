use ndarray::{Array4, Array2, ArrayView3, s};
use serde::{Deserialize};
use serde_json::Value;
use std::fs;
use image::{io::Reader as ImageReader, ImageFormat};


#[derive(Deserialize, Debug)]
struct Weight {
    shape: Vec<usize>,
    data: Value, 
}

fn flatten_data(data: &Value) -> Vec<f64> {
    fn flatten(value: &Value, out: &mut Vec<f64>) {
        match value {
            Value::Array(arr) => {
                for elem in arr {
                    flatten(elem, out);
                }
            }
            Value::Number(num) => {
                if let Some(f) = num.as_f64() {
                    out.push(f as f64);
                }
            }
            _ => panic!("Unexpected data type"),
        }
    }

    let mut result = Vec::new();
    flatten(data, &mut result);
    result
}


#[derive(Debug)]
struct Conv2DLayer {
    filters: Array4<f64>, // Форма (fh, fw, in_channels, out_channels)
    bias: Array2<f64>,    // Форма (out_channels, 1)
    input_shape: (usize, usize, usize),
}

impl Conv2DLayer {
    fn apply(&self, input: &Array4<f64>) -> Array4<f64> {
        let (n, h_in, w_in, c_in) = input.dim();
        let (fh, fw, _, fc) = self.filters.dim();
        
        let pad_h = (fh - 1) / 2;
        let pad_w = (fw - 1) / 2;
        
        let padded_input = pad_array(input.view(), pad_h, pad_w);
        
        let filters_2d = self.filters.view().into_shape((fh * fw * c_in, fc)).unwrap();
        let bias_1d = self.bias.column(0).into_owned();
        
        let mut output = Array4::zeros((n, h_in, w_in, fc));
        
        for i in 0..n {

            let batch_view = padded_input.slice(s![i, .., .., ..]);
            let cols = im2col(&batch_view, fh, fw, h_in, w_in);
            
            let mut result = cols.dot(&filters_2d);
            result += &bias_1d;
            
            result.mapv_inplace(relu);
            let shaped = result.into_shape((h_in, w_in, fc)).unwrap();
            output.slice_mut(s![i, .., .., ..]).assign(&shaped);
        }
        
        output
    }
}


fn pad_array(input: ndarray::ArrayView4<f64>, pad_h: usize, pad_w: usize) -> Array4<f64> {
    let (n, h, w, c) = input.dim();
    let mut padded = Array4::zeros((n, h + 2 * pad_h, w + 2 * pad_w, c));
    padded.slice_mut(s![.., pad_h..h+pad_h, pad_w..w+pad_w, ..]).assign(&input);
    padded
}

fn im2col(input: &ArrayView3<f64>, fh: usize, fw: usize, h_out: usize, w_out: usize) -> Array2<f64> {
    let (h_padded, w_padded, c) = input.dim();
    let mut cols = Array2::zeros((h_out * w_out, fh * fw * c));
    
    for j in 0..h_out {
        for k in 0..w_out {
            let window = input.slice(s![j..j+fh, k..k+fw, ..]);
            cols.row_mut(j * w_out + k)
                .assign(&window.to_shape(fh * fw * c).unwrap().t());
        }
    }
    cols
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

#[derive(Debug)]
struct BatchNormalizationLayer {
    mean: Array2<f64>,
    variance: Array2<f64>,
    gamma: Array2<f64>,
    beta: Array2<f64>,
}

impl BatchNormalizationLayer {

    fn apply_array4(&self, input: &Array4<f64>) -> Array4<f64> {
        let variance_eps = 1e-7;
        let mut output = input.clone();
        let (n, h, w, c) = input.dim();

        for i in 0..n {
            for j in 0..h {
                for k in 0..w {
                    for l in 0..c {
                        let mean = self.mean[(l, 0)];
                        let variance = self.variance[(l, 0)];
                        let gamma = self.gamma[(l, 0)];
                        let beta = self.beta[(l, 0)];

                        if !variance.is_finite() || !gamma.is_finite() || !beta.is_finite() {
                            panic!("Invalid values: variance {}, gamma {}, beta {}", variance, gamma, beta);
                        }
                        let res = (input[(i, j, k, l)] - mean)
                        / (variance + variance_eps).sqrt()
                        * gamma
                        + beta;
                        if !res.is_finite() {
                            panic!("Invalid res {:?} in {:?}, mean {:?}, var {:?}, gamma {:?}, beta {:?}", res, input[(i, j, k, l)], mean, variance, gamma, beta);
                        }
                        output[(i, j, k, l)] = res;
                    }
                }
            }
        }

        output
    }

    fn apply_array2(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut output = input.clone();

        for i in 0..input.shape()[0] {
            for j in 0..input.shape()[1] {
                let mean = self.mean[(j, 0)];
                let variance = self.variance[(j, 0)];
                let gamma = self.gamma[(j, 0)];
                let beta = self.beta[(j, 0)];

                output[(i, j)] = (input[(i, j)] - mean) / variance.sqrt() * gamma + beta;
            }
        }

        output
    }
}

#[derive(Debug)]
struct MaxPooling2DLayer {
    pool_size: (usize, usize),
}

impl MaxPooling2DLayer {
    fn apply(&self, input: &Array4<f64>) -> Array4<f64> {

        let (n, h, w, c) = input.dim();
        let (ph, pw) = self.pool_size;

        let out_h = h / ph;
        let out_w = w / pw;

        let mut output = Array4::<f64>::zeros((n, out_h, out_w, c));

        for i in 0..n {
            for j in 0..out_h {
                for k in 0..out_w {
                    for l in 0..c {

                        let start_h = j * ph;
                        let start_w = k * pw;
                        let end_h = start_h + ph;
                        let end_w = start_w + pw;

                        let mut max_val = f64::NEG_INFINITY;
                        for hh in start_h..end_h {
                            for ww in start_w..end_w {
                                max_val = max_val.max(input[(i, hh, ww, l)]);
                            }
                        }

                        output[(i, j, k, l)] = max_val;
                    }
                }
            }
        }

        output
    }
}


#[derive(Debug)]
struct FlattenLayer;

impl FlattenLayer {
    fn apply(&self, input: &Array4<f64>) -> Array2<f64> {
        let shape = input.shape();
        let flattened = input.iter().cloned().collect::<Vec<f64>>();
        Array2::from_shape_vec((flattened.len(), 1), flattened).unwrap()
    }
}

#[derive(Debug)]
struct DenseLayer {
    weights: Array2<f64>,
    bias: Array2<f64>,
}

impl DenseLayer {
    fn apply(&self, input: &Array2<f64>, relu: bool, sigmoid: bool) -> Array2<f64> {
        let mut result = input.dot(&self.weights);

        result += &self.bias.t();

        if relu {
            result.mapv_inplace(|x| if x > 0.0 { x } else { 0.0 });
        }
        if sigmoid {
            result.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
        }

        result
    }
}

#[derive(Debug)]
struct Model {
    conv1: Conv2DLayer,
    batch_norm1: BatchNormalizationLayer,
    pool1: MaxPooling2DLayer,

    conv2: Conv2DLayer,
    batch_norm2: BatchNormalizationLayer,
    pool2: MaxPooling2DLayer,

    conv3: Conv2DLayer,
    batch_norm3: BatchNormalizationLayer,
    pool3: MaxPooling2DLayer,

    conv4: Conv2DLayer,
    batch_norm4: BatchNormalizationLayer,
    pool4: MaxPooling2DLayer,

    flatten: FlattenLayer,

    dense1: DenseLayer,
    batch_norm5: BatchNormalizationLayer,
    dense2: DenseLayer,
    batch_norm6: BatchNormalizationLayer,
    dense3: DenseLayer,
}

impl Model {

    fn predict(&self, input: &Array4<f64>) -> Array2<f64> {
        let x = self.conv1.apply(input);
        let x = self.batch_norm1.apply_array4(&x);
        let x = self.pool1.apply(&x);
        let x = self.conv2.apply(&x);
        let x = self.batch_norm2.apply_array4(&x);
        let x = self.pool2.apply(&x);
        let x = self.conv3.apply(&x);
        let x = self.batch_norm3.apply_array4(&x);
        let x = self.pool3.apply(&x);
        let x = self.conv4.apply(&x);
        let x = self.batch_norm4.apply_array4(&x);
        let x = self.pool4.apply(&x);

        let flattened = self.flatten.apply(&x);

        let x = self.dense1.apply(&flattened.t().to_owned(), true, false);
        let x = self.batch_norm5.apply_array2(&x); 
        let x = self.dense2.apply(&x, true, false);  
        let x = self.batch_norm6.apply_array2(&x); 
        let x = self.dense3.apply(&x, false, true);  


        x
    }
}

fn load_weights() -> Vec<Weight> {
    let data = fs::read_to_string("weights-new-5.json").expect("Unable to read file");
    serde_json::from_str(&data).expect("Error parsing JSON")
}

fn build_model(weights: &Vec<Weight>) -> Model {
    let conv1_filters = Array4::from_shape_vec((3, 3, 3, 64), flatten_data(&weights[0].data)).unwrap();
    let conv1_bias = Array2::from_shape_vec((64, 1), flatten_data(&weights[1].data)).unwrap();

    let batch_norm1_gamma = Array2::from_shape_vec((64, 1), flatten_data(&weights[2].data)).unwrap();
    let batch_norm1_beta = Array2::from_shape_vec((64, 1), flatten_data(&weights[3].data)).unwrap();
    let batch_norm1_mean = Array2::from_shape_vec((64, 1), flatten_data(&weights[4].data)).unwrap();
    let batch_norm1_variance = Array2::from_shape_vec((64, 1), flatten_data(&weights[5].data)).unwrap();

    let conv2_filters = Array4::from_shape_vec((3, 3, 64, 128), flatten_data(&weights[6].data)).unwrap();
    let conv2_bias = Array2::from_shape_vec((128, 1), flatten_data(&weights[7].data)).unwrap();

    let batch_norm2_gamma = Array2::from_shape_vec((128, 1), flatten_data(&weights[8].data)).unwrap();
    let batch_norm2_beta = Array2::from_shape_vec((128, 1), flatten_data(&weights[9].data)).unwrap();
    let batch_norm2_mean = Array2::from_shape_vec((128, 1), flatten_data(&weights[10].data)).unwrap();
    let batch_norm2_variance = Array2::from_shape_vec((128, 1), flatten_data(&weights[11].data)).unwrap();

    let conv3_filters = Array4::from_shape_vec((3, 3, 128, 256), flatten_data(&weights[12].data)).unwrap();
    let conv3_bias = Array2::from_shape_vec((256, 1), flatten_data(&weights[13].data)).unwrap();

    let batch_norm3_gamma = Array2::from_shape_vec((256, 1), flatten_data(&weights[14].data)).unwrap();
    let batch_norm3_beta = Array2::from_shape_vec((256, 1), flatten_data(&weights[15].data)).unwrap();
    let batch_norm3_mean = Array2::from_shape_vec((256, 1), flatten_data(&weights[16].data)).unwrap();
    let batch_norm3_variance = Array2::from_shape_vec((256, 1), flatten_data(&weights[17].data)).unwrap();

    let conv4_filters = Array4::from_shape_vec((3, 3, 256, 512), flatten_data(&weights[18].data)).unwrap();
    let conv4_bias = Array2::from_shape_vec((512, 1), flatten_data(&weights[19].data)).unwrap();

    let batch_norm4_gamma = Array2::from_shape_vec((512, 1), flatten_data(&weights[20].data)).unwrap();
    let batch_norm4_beta = Array2::from_shape_vec((512, 1), flatten_data(&weights[21].data)).unwrap();
    let batch_norm4_mean = Array2::from_shape_vec((512, 1), flatten_data(&weights[22].data)).unwrap();
    let batch_norm4_variance = Array2::from_shape_vec((512, 1), flatten_data(&weights[23].data)).unwrap();
    let dense1_weights = Array2::from_shape_vec((32768, 256), flatten_data(&weights[24].data)).unwrap();
    let dense1_bias = Array2::from_shape_vec((256, 1), flatten_data(&weights[25].data)).unwrap();

    let batch_norm5_gamma = Array2::from_shape_vec((256, 1), flatten_data(&weights[26].data)).unwrap();
    let batch_norm5_beta = Array2::from_shape_vec((256, 1), flatten_data(&weights[27].data)).unwrap();
    let batch_norm5_mean = Array2::from_shape_vec((256, 1), flatten_data(&weights[28].data)).unwrap();
    let batch_norm5_variance = Array2::from_shape_vec((256, 1), flatten_data(&weights[29].data)).unwrap();

    let dense2_weights = Array2::from_shape_vec((256, 128), flatten_data(&weights[30].data)).unwrap();
    let dense2_bias = Array2::from_shape_vec((128, 1), flatten_data(&weights[31].data)).unwrap();

    let batch_norm6_gamma = Array2::from_shape_vec((128, 1), flatten_data(&weights[32].data)).unwrap();
    let batch_norm6_beta = Array2::from_shape_vec((128, 1), flatten_data(&weights[33].data)).unwrap();
    let batch_norm6_mean = Array2::from_shape_vec((128, 1), flatten_data(&weights[34].data)).unwrap();
    let batch_norm6_variance = Array2::from_shape_vec((128, 1), flatten_data(&weights[35].data)).unwrap();

    let dense3_weights = Array2::from_shape_vec((128, 1), flatten_data(&weights[36].data)).unwrap();
    let dense3_bias = Array2::from_shape_vec((1, 1), flatten_data(&weights[37].data)).unwrap();

    Model {
        conv1: Conv2DLayer {
            filters: conv1_filters,
            bias: conv1_bias,
            input_shape: (128, 128, 3),
        },
        batch_norm1: BatchNormalizationLayer {
            mean: batch_norm1_mean,
            variance: batch_norm1_variance,
            gamma: batch_norm1_gamma,
            beta: batch_norm1_beta,
        },
        pool1: MaxPooling2DLayer { pool_size: (2, 2) },

        conv2: Conv2DLayer {
            filters: conv2_filters,
            bias: conv2_bias,
            input_shape: (64, 64, 128),
        },
        batch_norm2: BatchNormalizationLayer {
            mean: batch_norm2_mean,
            variance: batch_norm2_variance,
            gamma: batch_norm2_gamma,
            beta: batch_norm2_beta,
        },
        pool2: MaxPooling2DLayer { pool_size: (2, 2) },

        conv3: Conv2DLayer {
            filters: conv3_filters,
            bias: conv3_bias,
            input_shape: (32, 32, 256),
        },
        batch_norm3: BatchNormalizationLayer {
            mean: batch_norm3_mean,
            variance: batch_norm3_variance,
            gamma: batch_norm3_gamma,
            beta: batch_norm3_beta,
        },
        pool3: MaxPooling2DLayer { pool_size: (2, 2) },

        conv4: Conv2DLayer {
            filters: conv4_filters,
            bias: conv4_bias,
            input_shape: (16, 16, 256),
        },
        batch_norm4: BatchNormalizationLayer {
            mean: batch_norm4_mean,
            variance: batch_norm4_variance,
            gamma: batch_norm4_gamma,
            beta: batch_norm4_beta,
        },
        pool4: MaxPooling2DLayer { pool_size: (2, 2) },

        flatten: FlattenLayer,

        dense1: DenseLayer {
            weights: dense1_weights,
            bias: dense1_bias,
        },
        batch_norm5: BatchNormalizationLayer {
            mean: batch_norm5_mean,
            variance: batch_norm5_variance,
            gamma: batch_norm5_gamma,
            beta: batch_norm5_beta,
        },
        dense2: DenseLayer {
            weights: dense2_weights,
            bias: dense2_bias,
        },
        batch_norm6: BatchNormalizationLayer {
            mean: batch_norm6_mean,
            variance: batch_norm6_variance,
            gamma: batch_norm6_gamma,
            beta: batch_norm6_beta,
        },
        dense3: DenseLayer {
            weights: dense3_weights,
            bias: dense3_bias,
        },
    }
}

fn load_and_preprocess_image(path: &str) -> Array4<f64> {

    let img = ImageReader::open(path)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image");

    // Изменяем размер до 128x128
    // let resized_img = img.resize_exact(128, 128, image::imageops::FilterType::Nearest);
    // let resized_img = img.resize_exact(128, 128, image::imageops::FilterType::Lanczos3);
    let resized_img = img.resize_exact(128, 128, image::imageops::FilterType::Gaussian);

    let resized_img = resized_img.to_rgba8();
    resized_img
        .save_with_format("resized_output.png", ImageFormat::Png)
        .expect("Failed to save image");

    let mut array = Array4::<f64>::zeros((1, 128, 128, 3)); // Размер: 1 x 128 x 128 x 3 (для модели)


    for (x, y, pixel) in resized_img.enumerate_pixels() {
        let [r, g, b, _] = pixel.0;
        array[[0, y as usize, x as usize, 0]] = r as f64 / 255.0;
        array[[0, y as usize, x as usize, 1]] = g as f64 / 255.0;
        array[[0, y as usize, x as usize, 2]] = b as f64 / 255.0;
    }

    array
}


fn main() {
    println!("Load weights");
    let weights = load_weights();
    println!("Build_model");
    let model = build_model(&weights);
    let path_to_images_folder = "images/";
    let image_path = "1.jpg";
    let image_path = "10.jpg";
    let image_path = "10014.jpg";
    let image_path = "1002.jpg";
    let image_path = "10006.jpg";
    let image_path = "tabyret.jpg";
    // let image_path = "monkey.jpg";
    // let image_path = "cat-2.jpg";
    // let image_path = "cat-1.jpg";
    // let image_path = "cat-4.png"; // - the photo is not square
    // let image_path = "cat-4-1.png"; // + the photo is square
    // let image_path = "cat-5.png";
    // let image_path = "street.png";
    // let image_path = "flower.png";
    let image_path = "fox.png"; // -
    // let image_path = "house.png";
    // let image_path = "cat-6.png";
    // let image_path = "cat-7.png"; // - 
    // let image_path = "cat-8.png"; 
    // let image_path = "cat-9.png"; 
    // let image_path = "car.png"; 
    // let image_path = "cat-10.png"; 
    // let image_path = "cat-11.png"; 
    // let image_path = "cat-12.png"; 
    // let image_path = "cat-13.png"; 
    // let image_path = "cats-3.jpg";
    // let image_path = "cats-a.jpg";
    // let image_path = "cat-15-2.jpg";
    // let image_path = "ai-cats.jpg";
    // let image_path = "ai-cats-2.jpg";
    // let image_path = "cats.png";
    // let image_path = "cat-15-4.jpg";
    // let image_path = "images.jpg";
    // let image_path = "cat-16.jpg";
    // let image_path = "cat-14.png"; // It's not a cat: 99.8811018907867% accuracy <- think that's the right answer

    println!("Load and preprocess image");
    let input_data = load_and_preprocess_image(&(path_to_images_folder.to_owned()+image_path));
    println!("Waiting predict...");
    let output = model.predict(&input_data);

    if let Some(res) = output.into_flat().get(0) {
        let res = res * 100 as f64;
        if res > 50.0 {
            println!("\n It's a cat: {:?}% accuracy \n", res);
        } else {
            println!("\n It's NOT a cat: {:?}% accuracy \n", 100.0 - res);
        }

    }
    
}




