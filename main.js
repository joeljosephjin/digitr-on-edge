const ort = require('onnxruntime-web/webgpu');

const MAX_WIDTH = 28;
const MAX_HEIGHT = 28;
const MODEL_WIDTH = 28;
const MODEL_HEIGHT = 28;

const MODEL_MAP = {
    mnist_model: ["models/mnist_cnn.onnx"],
};

const config = getConfig();

ort.env.wasm.numThreads = config.threads;
ort.env.wasm.proxy = true;

let canvas;
let filein;
let decoder_latency;

var image_embeddings;
var sess = [];
var imageImageData;

function log(i) {
    document.getElementById('status').innerHTML += `<br>[${performance.now().toFixed(3)}] ` + i;
    console.log(i);
}


/**
 * get some parameters from url
 */
function getConfig() {
    const query = window.location.search.substring(1);
    var config = {
        model: "mnist_model",
        provider: "webgpu",
        device: "gpu",
        threads: "1",
    };
    let vars = query.split("&");
    for (var i = 0; i < vars.length; i++) {
        let pair = vars[i].split("=");
        if (pair[0] in config) {
            config[pair[0]] = decodeURIComponent(pair[1]);
        } else if (pair[0].length > 0) {
            throw new Error("unknown argument: " + pair[0]);
        }
    }
    config.threads = parseInt(config.threads);
    return config;
}

async function resizeTensor(originalTensor) {
    // Assuming 'originalTensor' is your ONNX Runtime Web tensor with dimensions [1, 3, 28, 28]
    // const originalShape = originalTensor.dims; // Get the original tensor's dimensions

    // Calculate the total number of elements in the tensor
    // const totalElements = originalShape.reduce((acc, val) => acc * val);

    // Assuming 'originalData' is your Float32Array containing the tensor's data
    const originalData = originalTensor.data;

    // Reshape the tensor to [1, 1, 28, 28]
    const reshapedTensor = new ort.Tensor(originalData.slice(0, 28*28), [1, 1, 28, 28]); // Adjust the new shape as needed

    // 'reshapedTensor' now contains the data from 'originalTensor' but with the new shape [1, 1, 28, 28]
    log(`[resizeTensor] reshapedTensor is ${reshapedTensor.dims}`);
    return reshapedTensor;
}

async function multiplyTensor(tensor, scalar) {
    // Assuming 'tensor' is your ONNX Runtime Web tensor
    const data = tensor.data;

    // Multiply every element by 255
    for (let i = 0; i < data.length; i++) {
        data[i] *= scalar;
    };

    return tensor;
}

async function getKeysAndValues(originalObject) {
    for (key in originalObject) {
        log(`[getKeysAndValues] object[${key}]: ${originalObject[key]}`);
    }
    return originalObject[key];
}

async function ifWhiteBackground(imgArr, threshold = 0.3) {
    // Assuming 'imgArr' is a JavaScript array or typed array representing image data
    // const backgroundValue = 255; // Value representing the background color
    const backgroundValue = 1; // Value representing the background color

    // Count the number of elements matching the background value
    var count = 0;
    // Assuming 'tensor' is your ONNX Runtime Web tensor
    const data = imgArr.data;

    for (let i = 0; i < data.length; i++) {
        // break;
        if (data[i] == backgroundValue) {
            // log(`data ${i} is ${data[i]}`);
            count = count + 1;
        }
    };

    // Calculate the percentage of background pixels
    // const percent = count / imgArr.length;
    var percent = count / 784;

    log(`[ifWhiteBackground] percent whitenesss is ${percent}; count is ${count}; ${imgArr.dims}`)

    // Check if the percentage exceeds the threshold
    if (percent >= threshold) {
        return true;
    } else {
        return false;
    }
}

async function invertImage(imgArr) {
    const data = imgArr.data;

    for (let i = 0; i < data.length; i++) {
        data[i] = 1 - data[i];
    };

    return imgArr;
}

async function getImageDataFromTensor(imageTensor, width, height) {
    // Convert the tensor data to a regular array (Assuming it's a Float32Array or similar)
    tensorData = Array.from(imageTensor.data);
    tensorData = tensorData.map(value => value * 255);

    log(`[getImageDataFromTensor] tensorData.length: ${tensorData.length}`)

    const grayscaleData = new Uint8ClampedArray(tensorData);

    const rgbaData = new Uint8ClampedArray(width * height * 4); // RGBA pixel data

    // Populate the RGBA pixel data with the grayscale values
    for (let i = 0; i < width * height; i++) {
        const grayscaleValue = grayscaleData[i];

        // Set the same grayscale value for RGBA channels (R, G, B, A)
        const pixelIndex = i * 4;
        rgbaData[pixelIndex] = grayscaleValue; // Red channel
        rgbaData[pixelIndex + 1] = grayscaleValue; // Green channel
        rgbaData[pixelIndex + 2] = grayscaleValue; // Blue channel
        rgbaData[pixelIndex + 3] = 255; // Alpha channel (fully opaque)
    }

    // Create an ImageData object from the RGBA pixel data
    const imageData = new ImageData(rgbaData, width, height);

    log(`[getImageDataFromTensor] imageData: ${imageData}###`)

    // Use imageData with a canvas, putImageData(), or other canvas operations
    return imageData;
}

async function htmlDisplayImage(imageTensor, elementID, width=28, height=28) {
    // width = imageTensor.dims[2];
    // height = imageTensor.dims[3];

    log(`[htmlDisplayImage] imageTensor: ${width}, ${height}###`);

    const imageData = await getImageDataFromTensor(imageTensor, width, height);

    // Create a canvas and draw the image data onto it
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = width;
    canvas.height = height;
    ctx.putImageData(imageData, 0, 0);

    // Convert the canvas content to a data URL
    const dataURL = canvas.toDataURL('image/png'); // Change 'image/png' to the desired image format

    const imgElement = document.getElementById(elementID);
    imgElement.src = dataURL;
}

/*
* invert a white background to black or remain so if black
*/
async function invertWhiteBackground(imgArr) {


    isWhite = await ifWhiteBackground(imgArr);
    // isWhite = true;
    if (isWhite == true) {
        // invert image
        log(`[invertWhiteBackground] yaay! image is white background; gotta invert!`);
        imgArr = await invertImage(imgArr);
        log(`[invertWhiteBackground] yaay! image's white background inverted successfully!`);
        return imgArr;
    } else {
        log(`[invertWhiteBackground] noo! img is NOT white background`);
        return imgArr;
    }
}

async function imagePreprocessing(imgArr) {

    // 1, 1, 28, 28
    imgArr = await ort.Tensor.fromImage(imgArr, options = {resizedWidth: MODEL_WIDTH, resizedHeight: MODEL_HEIGHT});

    // abc = await ifWhiteBackground(imgArr);

    imgArr = await resizeTensor(imgArr);

    // find_white_background
    imgArr = await invertWhiteBackground(imgArr);
    // bitwise_not
    // await htmlDisplayImage(imgArr, 'debug-image'); // for DEBUGGING

    return imgArr; 

}

/**
 * handler called when image available
 * run the model (encoder) on the image
 */
async function modelForward(img) {
    // await htmlDisplayImage(img, 'debug-image'); // for DEBUGGING
    // await htmlDisplayImage(img, 'debug-image'); // for DEBUGGING

    const prediction_element = document.getElementById("prediction-element");
    prediction_element.innerHTML = "?";
    
    filein.disabled = true;
    const canvas = document.createElement('canvas');
    // decoder_latency.innerText = "";
    canvas.style.cursor = "wait";
    image_embeddings = undefined;
    var width = img.width;
    var height = img.height;

    if (width > height) {
        if (width > MAX_WIDTH) {
            height = height * (MAX_WIDTH / width);
            width = MAX_WIDTH;
        }
    } else {
        if (height > MAX_HEIGHT) {
            width = width * (MAX_HEIGHT / height);
            height = MAX_HEIGHT;
        }
    }
    width = Math.round(width);
    height = Math.round(height);

    canvas.width = width;
    canvas.height = height;
    var ctx = canvas.getContext("2d");
    log(`[modelForward] img.height: ${img.height}### img.width: ${img.width}`)
    ctx.drawImage(img, 0, 0, width, height);

    imageImageData = ctx.getImageData(0, 0, width, height);

    log(`[modelForward] before await imagePreprocessing...`);
    const imgTensor = await imagePreprocessing(imageImageData);
    log(`[modelForward] after await imagePreprocessing...`);

    // await htmlDisplayImage(imgTensor, 'debug-image'); // for DEBUGGING

    log(`[modelForward] ###imgTensor.dims is this: ${imgTensor.dims}###`);
    log(`[modelForward] ###imgTensor sample is this: ${imgTensor.data.slice(110, 119)}###`);
    
    const feed = { "input.1": imgTensor };
    const s = await sess[0];

    const start = performance.now();
    log("[modelForward] s:", s)
    log(`[modelForward] forward passing image through the model`);
    image_embeddings = await s.run(feed); // MODEL.FORWARD
    log(`[modelForward] forward passing completed!`);

    const emb_property = await getKeysAndValues(image_embeddings);
    probs = emb_property['cpuData'];
    max_prob = Math.max(...probs);
    maxInd = probs.indexOf(max_prob);

    probsArray = Array.from(probs)
    expProbNums = probsArray.map(function(each_element){
        return Number(Math.exp(each_element));
    });
    totalProbNums = expProbNums.reduce((a, b) => a + b, 0);
    expProbProbs = expProbNums.map(function(each_element){
        return Number(((each_element/totalProbNums)*100).toFixed(2));
    });
    prediction_element.innerHTML = `<div class="tooltip_custom">${maxInd}<span class="tooltiptext_custom">Probabilities: ${expProbProbs.join("%, ")}%</span></div>`;
    log(`[modelForward] Class: ${maxInd} |  Probabilities: ${expProbProbs.join("%, ")}`);

    filein.disabled = false;
}


/**
 * fetch and cache url
Uses functions:
1. log
 */
async function fetchAndCache(url) {
    try {
        const cache = await caches.open("onnx");

        if (config.clear_cache) {cache.delete(url);}

        let cachedResponse = await cache.match(url);
        if (cachedResponse == undefined) {
            await cache.add(url);
            cachedResponse = await cache.match(url);
            log(`[fetchAndCache] ${url} loaded (from network)`);
        } else {
            log(`[fetchAndCache] ${url} loaded (from cache)`);
        }
        const data = await cachedResponse.arrayBuffer();
        return data;
    } catch (error) {
        log(`[fetchAndCache] ${url} (from network)`);
        return await fetch(url).then(response => response.arrayBuffer());
    }
}

/*
 * load encoder and decoder sequentially
Uses functions:
1. fetchAndCache
2. modelForward (model forward)
3. log
 */
async function load_model(model, img) {
    idx=0;
    log(`[load_model] idx: ${idx}`);

    // --- select device webnn or webgpu
    let provider = config.provider;
    switch (provider) {
        case "webnn":
            if (!("ml" in navigator)) {
                throw new Error("webnn is NOT supported");
            }
            provider = {
                name: "webnn",
                deviceType: config.device,
                powerPreference: 'default'
            };
            log('[load_model] webnn activated!');
            break;
        case "webgpu":
            if (!navigator.gpu) {
                throw new Error("webgpu is NOT supported");
            }
            log('[load_model] webgpu activated!');
            break;
    }
    const opt = { executionProviders: [provider] };
    // --- select device webnn or webgpu

    fetchAndCache(model[idx]).then((data) => {
        sess[idx] = ort.InferenceSession.create(data, opt); // data and device

        // -- useless garbage
        sess[idx].then(() => {
            log(`[load_model] [fetchAndCache.then] ${model[idx]} successfully loaded!`);
        }, (e) => {
            log(`[load_model] [fetchAndCache.then] ${model[idx]} load failed with ${e}.`);
            throw e;
        });

        if (img !== undefined) {
            log("[load_model] [fetchAndCache.then] img is defined!");
            // modelForward(img);
        }
        else {log("[load_model] [fetchAndCache.then] img is undefined!")}
    })
}




// ----------------------------------

shouldCamOn = false;
var captureInterval;

function captureCameraImage() {
  var video = document.querySelector('video');

  var img = document.getElementById('original-image');

  // Capture an image from the video stream
  var canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  var context = canvas.getContext('2d');
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Set the captured image as the src of the img element
  img.src = canvas.toDataURL('image/png');
//   img.style.display = "inline";
  console.log(`[captureCameraImage] img.src rewritten!`);

  modelForward(img);
  console.log(`[captureCameraImage] model forward done!`);
}

function switchCamera() {
  shouldCamOn = !shouldCamOn;

  if (shouldCamOn==true) {
    onCamera();
  } else {
    offCamera();
  }
}

function offCamera() {
  console.log(`[offCamera] button pressed... shouldCamOn: ${shouldCamOn}`);

  var videoElement = document.getElementById('video-element');

  var stream = videoElement.srcObject;
    var tracks = stream.getTracks();
    tracks.forEach(function(track) {
        track.stop();
    });

    videoElement.remove();
  console.log(`[offCamera] camera_div removed...`);

  var originalImage = document.getElementById('original-image');
  originalImage.style.display = "inline";
  console.log(`[offCamera] originalImage recovered...`);

  let img = document.getElementById("original-image");
  modelForward(img);
  console.log(`[offCamera] originalImage prediction complete...`);

  if (captureInterval) {
    clearInterval(captureInterval);
  }
}

function onCamera() {
  console.log(`[onCamera] button pressed... shouldCamOn: ${shouldCamOn}`);
  // Get the div element
  var divElement = document.getElementById('camera_div');
  
  var originalImage = document.getElementById('original-image');
  originalImage.style.display = "none";

  // Check if getUserMedia is available
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      // Request access to the camera
      navigator.mediaDevices.getUserMedia({ video: true })
          .then(function(stream) {
              // Create a video element
              var videoElement = document.createElement('video');
              videoElement.id = "video-element";
              // Set the source of the video element to the camera stream
              videoElement.srcObject = stream;
              // Autoplay the video
              videoElement.autoplay = true;
              // Set the width of the video to match the container
              videoElement.style.width = '100%';
              videoElement.style.height = '100%';
              // Append the video element to the div
              divElement.appendChild(videoElement);

              captureInterval = setInterval(captureCameraImage, 3000);
          })
          .catch(function(error) {
              console.error('Error accessing the camera:', error);
          });
  } else {
      console.error('getUserMedia is not supported on this browser');
  }
}
// ---------------------------------------

/*
uses the functions:
1. load_model
2. modelForward (encoder)
3. handleClick (decoder)
4. log
*/
async function main() {

    const model = MODEL_MAP[config.model]; // load the model

    log(`[main] config.model: ${config.model}`);

    filein = document.getElementById("file-in");

    let img = document.getElementById("original-image"); // egyptian-cat

    log("[main] BEFORE model loading...")
    load_model(model, img).then(() => {}, (e) => {log(e);}); // [EXT_FUNC] load_model
    log("[main] model loaded; sess:", sess)

    log("[main] [special] before lone modelForward...")
    await new Promise(r => setTimeout(r, 2000));
    modelForward(img);
    log("[main] [special] after lone modelForward...")

    // --- [2] image upload
    img.onload = function () {
        log(`[main] [img.onload] before model forwarding image...`);
        modelForward(img);
        log(`[main] [img.onload] after model forwarding image...`);
    }

    filein.onchange = function (evt) {
        log(`[main] [filein.onchange] beginning of filein.onchange ...`);
        let target = evt.target || window.event.src;
        let files = target.files;
        if (FileReader && files && files.length) {
            log(`[main] [filein.onchange] [if (FileReader &&] before new filereader`);
            let fileReader = new FileReader();
            log(`[main] [filein.onchange] [if (FileReader &&] after new filereader`);
            fileReader.onload = () => {
                log(`[main] [filein.onchange] [if (FileReader &&] [fileReader.onload] before img.src = fileReader.result`);
                img.src = fileReader.result;
                log(`[main] [filein.onchange] [if (FileReader &&] [fileReader.onload] after img.src = fileReader.result`);
            }
            fileReader.readAsDataURL(files[0]);
        }
    };

    // --- use_camera
    cameraElement = document.getElementById("use_camera");
    cameraElement.onchange = function (evt) {
        switchCamera();
    }

}


document.addEventListener("DOMContentLoaded", () => { main(); });
