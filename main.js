const ort = require('onnxruntime-web/webgpu');

const MAX_WIDTH = 28;
const MAX_HEIGHT = 28;
const MODEL_WIDTH = 28;
const MODEL_HEIGHT = 28;

const MODEL_MAP = {
    sam_b: ["models/mnist_cnn.onnx"],
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
        model: "sam_b",
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
    log(`reshapedTensor is ${reshapedTensor.dims}`);
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
        log(`object[${key}]: ${originalObject[key]}`);
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

    log(`percent whitenesss is ${percent}; count is ${count}; ${imgArr.dims}`)

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

    console.log(`[htmlDisplayImage] tensorData.length: ${tensorData.length}`)

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

    console.log(`[htmlDisplayImage] imageData: ${imageData}###`)

    // Use imageData with a canvas, putImageData(), or other canvas operations
    return imageData;
}

async function htmlDisplayImage(imageTensor, elementID, width=28, height=28) {
    // width = imageTensor.dims[2];
    // height = imageTensor.dims[3];

    console.log(`[htmlDisplayImage] imageTensor: ${width}, ${height}###`);

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
        log(`yaay! image is white background; gotta invert!`);
        imgArr = await invertImage(imgArr);
        return imgArr;
    } else {
        log(`noo! img is NOT white background`);
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
    // prediction_element.innerText = "";
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
    console.log(`[modelForward] img.height: ${img.height}### img.width: ${img.width}`)
    ctx.drawImage(img, 0, 0, width, height);

    imageImageData = ctx.getImageData(0, 0, width, height);

    const imgTensor = await imagePreprocessing(imageImageData);

    // await htmlDisplayImage(imgTensor, 'debug-image'); // for DEBUGGING

    log(`###imgTensor.dims is this: ${imgTensor.dims}###`);
    log(`###imgTensor sample is this: ${imgTensor.data.slice(110, 119)}###`);
    
    const feed = { "input.1": imgTensor };
    const s = await sess[0];

    const start = performance.now();
    console.log("[debug] s:", s)
    image_embeddings = await s.run(feed); // MODEL.FORWARD

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
    prediction_element.innerHTML = `<div class="tooltip_custom">Class: ${maxInd}<span class="tooltiptext_custom">Probabilities: ${expProbProbs.join("%, ")}%</span></div>`;

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
            log(`${url} (from network)`);
        } else {
            log(`${url} (from cache)`);
        }
        const data = await cachedResponse.arrayBuffer();
        return data;
    } catch (error) {
        log(`${url} (from network)`);
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
async function load_model(model, idx, img) {
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
            break;
        case "webgpu":
            if (!navigator.gpu) {
                throw new Error("webgpu is NOT supported");
            }
            break;
    }
    const opt = { executionProviders: [provider] };
    // --- select device webnn or webgpu

    log(`[load_model] idx is what? ${idx}`)

    fetchAndCache(model[idx]).then((data) => {
        sess[idx] = ort.InferenceSession.create(data, opt); // data and device

        // -- useless garbage
        sess[idx].then(() => {
            log(`${model[idx]} loaded. yoy`);
            if (idx == 0) {
                log("[load_model] idx is 0; so going for next recursion of model");
                // load_model(model, 1); // recursive load_model ?
            }
        }, (e) => {
            log(`${model[idx]} failed with ${e}.`);
            throw e;
        });
        if (img !== undefined) {
            log("img is undefined!");
            // modelForward(img);
        }
        else {log("img is not undefined!")}
        // -- useless garbage

    })
}

/*
uses the functions:
1. load_model
2. modelForward (encoder)
3. handleClick (decoder)
4. log
*/
async function main() {
    const model = MODEL_MAP[config.model]; // load the model (encoder or decoder) ?
    // --- [1] get the image from the user
    canvas = document.getElementById("img_canvas");
    // canvas.addEventListener("click", handleClick);
    // canvas.style.cursor = "wait";

    filein = document.getElementById("file-in");
    // decoder_latency = document.getElementById("decoder_latency");

    let img = document.getElementById("original-image");
    // --- [1] get the image from the user

    console.log("[debug] BEFORE model loading...")
    load_model(model, 0, img).then(() => {}, (e) => {log(e);}); // [EXT_FUNC] load_model
    console.log("[debug] model loaded; sess:", sess)
    // load_model(model, 0, img).then(() => {}, (e) => {}); // [EXT_FUNC] load_model

    // --- [2] image upload
    filein.onchange = function (evt) {
        let target = evt.target || window.event.src, files = target.files;
        if (FileReader && files && files.length) {
            let fileReader = new FileReader();
            fileReader.onload = () => {
                img.onload = () => modelForward(img); // [EXT_FUNC] modelForward
                img.src = fileReader.result;
            }
            fileReader.readAsDataURL(files[0]);
        }
    };
    // --- [2] image upload
}

document.addEventListener("DOMContentLoaded", () => { main(); });
