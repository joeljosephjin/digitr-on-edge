# Run Digit Recognizer in your browser using webgpu and onnxruntime-web

[Demo Site](http://13.53.147.31:8085/index.html?provider=wasm)

## About

This is a number recognizer. an enhancement of an MNIST classifier.

Why on the edge? Why not run on the backend?
- does not require internet; hence does not depend on network bandwidth, latency, etc...

Why use ONNX instead of PyTorch?
- PyTorch library is heavy requiring higher usage of RAM and disk space.




## how to develop

`sudo npx light-server -s . -p 8888` (this is fully enough ig)

`npm run watch` (garbage)

`npm run build`


## plan

- cleanup the UI (in progress)
    - remove the latency part (in progress)
- add option for live camera feed
- change the names in readme
- make wasm the default
- do new UI















## old stuff

This example demonstrates how to run [Segment-Anything](https://github.com/facebookresearch/segment-anything) in your 
browser using [onnxruntime-web](https://github.com/microsoft/onnxruntime) and webgpu.

Segment-Anything is a encoder/decoder model. The encoder creates embeddings and using the embeddings the decoder creates the segmentation mask.

One can run the decoder in onnxruntime-web using WebAssembly with  latencies at ~200ms. 

The encoder is much more compute intensive and takes ~45sec using WebAssembly what is not practical.
Using webgpu we can speedup the encoder ~50 times and it becomes visible to run it inside the browser, even on a integrated GPU.

## Usage

### Installation
First, install the required dependencies by running the following command in your terminal:
```sh
npm install
```

### Build the code
Next, bundle the code using webpack by running:
```sh
npm run build
```
this generates the bundle file `./dist/bundle.min.js`

### Create an ONNX Model

We use [samexporter](https://github.com/vietanhdev/samexporter) to export encoder and decoder to onnx.
Install samexporter:
```sh
pip install https://github.com/vietanhdev/samexporter
```
Download the pytorch model from [Segment-Anything](https://github.com/facebookresearch/segment-anything). We use the smallest flavor (vit_b).
```sh
curl -o models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
Export both encoder and decoder to onnx:
```sh
python -m samexporter.export_encoder --checkpoint models/sam_vit_b_01ec64.pth \
    --output models/sam_vit_b_01ec64.encoder.onnx \
    --model-type vit_b 

python -m samexporter.export_decoder --checkpoint models/sam_vit_b_01ec64.pth \
    --output models/sam_vit_b_01ec64.decoder.onnx \
    --model-type vit_b \
    --return-single-mask
```
### Start a web server
Use NPM package `light-server` to serve the current folder at http://localhost:8888/.
To start the server, run:
```sh
npx light-server -s . -p 8888
```

### Point your browser at the web server
Once the web server is running, open your browser and navigate to http://localhost:8888/. 
You should now be able to run Segment-Anything in your browser.

## TODO
* add support for fp16
* add support for MobileSam





## stuff

###t sample is this: 0.9686274528503418,0.9647058844566345,0.9647058844566345,0.9647058844566345,0.9647058844566345### - 2.jpeg

encoder: Class: 7###Embeddings: -2.205502510070801,-2.6346242427825928,-2.0290040969848633,-1.9202640056610107,-3.2169065475463867,-2.229710102081299,-2.5746657848358154,-1.734506368637085,-3.0979933738708496,-2.3583126068115234###

encoder: Class: 7###Embeddings: -2.205502510070801,-2.6346242427825928,-2.0290040969848633,-1.9202640056610107,-3.2169065475463867,-2.229710102081299,-2.5746657848358154,-1.734506368637085,-3.0979933738708496,-2.3583126068115234###

encoder: Class: 7###Embeddings: -2.205502510070801,-2.6346242427825928,-2.0290040969848633,-1.9202640056610107,-3.2169065475463867,-2.229710102081299,-2.5746657848358154,-1.734506368637085,-3.0979933738708496,-2.3583126068115234###

------- 9.jpeg --------

encoder: Class: 7###Embeddings: -2.1698496341705322,-2.6524527072906494,-1.922837495803833,-1.8882012367248535,-3.354196071624756,-2.270970344543457,-2.5470893383026123,-1.7632544040679932,-3.152177333831787,-2.4408345222473145###

encoder: Class: 7###Embeddings: -2.1698496341705322,-2.6524527072906494,-1.922837495803833,-1.8882012367248535,-3.354196071624756,-2.270970344543457,-2.5470893383026123,-1.7632544040679932,-3.152177333831787,-2.4408345222473145###

-------- 2.jpeg ------

encoder: Class: 7###Probabilities: -2.1698496341705322,-2.6524527072906494,-1.922837495803833,-1.8882012367248535,-3.354196071624756,-2.270970344543457,-2.5470893383026123,-1.7632544040679932,-3.152177333831787,-2.4408345222473145###

encoder: Class: 7###Probabilities: -2.1698496341705322,-2.6524527072906494,-1.922837495803833,-1.8882012367248535,-3.354196071624756,-2.270970344543457,-2.5470893383026123,-1.7632544040679932,-3.152177333831787,-2.4408345222473145###

[78.500] ###t sample is this: 0.9686274528503418,0.9647058844566345,0.9647058844566345,0.9647058844566345,0.9647058844566345###



