package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"proj2/png"
	"runtime"
	"sync"
)

// Instructions for input args
func printUsage() {
	usage := "editor [-p=[number of threads]]\n" +
	"\t-p=[number of threads] = An optional flag to run the editor in its parallel version.\n" +
	"\t\tCall and pass the runtime.GOMAXPROCS(...) function the integer\n" +
	"\t\tspecified by [number of threads].\n"
	fmt.Printf("Usage: " + usage)
}

func main() {
	if len(os.Args) > 2 {
		printUsage()
		os.Exit(0)
	}
	numThreads := flag.Int("p", 0, "an int representing number of threads")
	flag.Parse()

	if *numThreads == 0 {
		processSequential()
	} else {
		processParallel(*numThreads)
	}
}

func processSequential(){
	imageTasks := readJSONInputTasks()
	for i:=0; i < len(imageTasks); i++{
		processTask(imageTasks[i])
	}
}

func processParallel(numThreads int){
	runtime.GOMAXPROCS(numThreads)
	numReaders := int(math.Ceil(float64(numThreads) * (1.0/5.0)))
	readerDone := make(chan bool)

	var readerMutex sync.Mutex // a lock to allow us to have multiple threads read from Stdin in thread safe manner
	blockSize := 2 //number of JSON tasks a reader should attempt to chunk and grab
	dec := json.NewDecoder(os.Stdin)

	for i := 0; i < numReaders; i++ {
		go reader(numThreads, blockSize, readerDone, &readerMutex, dec, i)
	}

	//wait until all readers are done using a channel
	for i := 0; i < numReaders; i++{
		<- readerDone
	}
}

// Reads in JSON tasks from Stdin and do any preparation needed before applying their effects. Each reader is associated
// with a single pipeline of workers
func reader(numThreads int, blockSize int, readerDone chan bool, mutex *sync.Mutex, dec *json.Decoder, readerId int){
	for true {
		imageTasksChannel := readJSONInputTasksParallel(mutex, blockSize, dec)
		numTasks := len(imageTasksChannel)
		if numTasks == 0 {
			readerDone <- true
			break
		}

		//every reader spawns a single worker pipeline goroutine
		workerDone := make(chan bool, 1)
		go worker(numThreads, numTasks, imageTasksChannel, workerDone)
		close(imageTasksChannel) //close out the imageTasksChannel once worker is done processing it

		//wait until worker goroutine finishes
		<- workerDone
	}
}

// Pipeline workers are in charge of performing the filtering effects. Each stage should be dedicated to a
// specific filtering effect
func worker(numThreads int, numTasks int, imageTasksChannel <- chan ImageTask, workerDone chan bool) {
	for taskCounter:=0; taskCounter < numTasks; taskCounter++ { //loop through the JSON tasks we took from Stdin, upper bounded by blockSize
		imageTask := <- imageTasksChannel
		effects := imageTask.Effects
		pngImg, err := png.Load(imageTask.InPath)
		if err != nil {
			panic(err)
		}

		//**BEGINNING OF PIPELINE SECTION**
		//pipeline workers using the take-and-repeat pipeline structure
		//where each effect must be applied in order and within each effect we perform data decomposition in parallel
		processEffectParallel := func(effectsDone <- chan interface{}, effects []string, effectsCounter *int, pngImg *png.Image) <- chan *png.Image {
			imgStream := make(chan *png.Image)
			go func() {
				defer close(imgStream)
				for i := 0; i < len(effects); i++{
					effect := effects[*effectsCounter]
					pngImg :=parallelDecomposeEffect(pngImg, effect, numThreads)

					//if we're not on the final effect, pass the in img to out img to stack effects
					if i != len(effects) -1 {
						pngImg.SetImgOutToIn()
					}
					select {
					case <-effectsDone:
						return
					case imgStream <- pngImg:
						*effectsCounter++
					}
				}
			}()
			return imgStream
		}

		pipelineEffects := func(effectsDone <- chan interface{}, imgStream <- chan *png.Image, numEffects int) <- chan *png.Image {
			takeImgStream := make(chan *png.Image)
			go func() {
				defer close(takeImgStream)
				for effectsCounter := 0; effectsCounter < numEffects; effectsCounter++{
					select {
					case <-effectsDone:
						return
					case takeImgStream <- <-imgStream:
					}
				}
			}()
			return takeImgStream
		}

		effectsDone := make(chan interface{})
		effectsCounter := new(int)
		*effectsCounter = 0
		for range pipelineEffects(effectsDone,
			processEffectParallel(effectsDone, effects, effectsCounter, pngImg),
			len(effects)){}
		close(effectsDone)
		// **END OF PIPELINE SECTION**

		//save image
		writerDone := make(chan bool, 1)
		go writer(pngImg, imageTask.OutPath, writerDone)
		<- writerDone //wait until writer goroutine finishes
	}
	workerDone <- true
}

// Writers save the filtered image to its outpath file
func writer(pngImg *png.Image, outPath string, writerDone chan bool){
	err := pngImg.Save(outPath)
	if err != nil {
		panic(err)
	}
	writerDone <- true
}

//spawns numThread number of goRoutines, which will decompose a single image and perform effect on horizontally sliced subimages in parallel
func parallelDecomposeEffect(pngImg *png.Image, effect string, numThreads int) *png.Image{
	subImageWaitChannel := make(chan bool)
	height := pngImg.GetHeight()
	sectionHeight := math.Ceil(float64(height) / float64(numThreads))

	for sectionIndex := 0; sectionIndex < numThreads; sectionIndex++ {
		floor := float64(sectionIndex) * sectionHeight + 1
		if sectionIndex == 0 {
			floor = float64(0)
		}
		ceil := float64(sectionIndex + 1) * sectionHeight
		go processPartialImg(subImageWaitChannel, pngImg, effect, floor, ceil)

	}

	//wait to make sure all subimages complete by emptying out channel
	for sectionIndex := 0; sectionIndex < numThreads; sectionIndex++ {
		<- subImageWaitChannel
	}
	return pngImg
}

func processPartialImg(subImageWaitChannel chan bool, pngImg *png.Image, effect string, floor float64, ceil float64) {
	subImg := png.NewImg(pngImg.GetSubImg(int(floor)-5, int(ceil)+5)) // need small buffers on floor and ceil so subimage can convolute on subimages' edges properly
	processEffect(subImg, effect)
	pngImg.UseSubsetImg(subImg, int(floor), int(ceil))
	subImageWaitChannel <- true
}

// Reads in Stdin JSON inputs in a thread safe manner by locking each time it's called. Reader goroutines will
// all attempt to access Stdin through this function. Outputs a channel of ImageTasks that gets passed downstream to
// worker goroutine
func readJSONInputTasksParallel(lock *sync.Mutex, blockSize int, dec *json.Decoder) chan ImageTask{
	lock.Lock()
	imageTasksChannel := make(chan ImageTask, blockSize)
	for i:=0; i < blockSize; i++{ //loop through blocksize amount of each json objects as ImageTask
		var t ImageTask
		err := dec.Decode(&t)
		if err != nil {
			if err == io.EOF{
				break
			}
		}
		imageTasksChannel <- t
	}
	lock.Unlock()
	return imageTasksChannel
}

// Reads in Stdin JSON inputs sequentially
func readJSONInputTasks() []ImageTask{
	var imageTasks []ImageTask
	dec := json.NewDecoder(os.Stdin)
	for { //loop through and process each json object as task
		var t ImageTask
		err := dec.Decode(&t)
		if err != nil {
			if err == io.EOF{
				break
			}
			fmt.Println(err)
		}
		imageTasks = append(imageTasks, t)
	}
	return imageTasks
}

// Sequentially execute each effect in order without image decomposition
func processTask(t ImageTask) {
	pngImg, err := png.Load(t.InPath)
	if err != nil {
		panic(err)
	}

	for i := 0; i < len(t.Effects); i++ {
		effect := t.Effects[i]
		processEffect(pngImg, effect)

		//if we're not on the final effect, pass the in img to out img to stack effects
		if i != len(t.Effects) - 1 {
			pngImg.SetImgOutToIn()
		}
	}
	err = pngImg.Save(t.OutPath)
	if err != nil {
		panic(err)
	}
}

// Based on the input effect command string, execute the effect on the image
func processEffect(pngImg *png.Image, effect string){
	if effect == "G"{
		pngImg.Grayscale()
	} else if effect == "S"{
		pngImg.Sharpen()
	} else if effect == "E"{
		pngImg.EdgeDetect()
	} else if effect == "B"{
		pngImg.Blur()
	} else {
		fmt.Println("WARNING: Effect command:", effect, " not recognized")
	}
}

// Each line from Stdin represents a JSON task which has an image's inpath, outputh, and an array of effects we want
type ImageTask struct {
	InPath string `json:"inPath"` // filepath of images to read in
	OutPath string `json:"outPath"`// filepath to save the image after applying effects
	Effects []string `json:"effects"`// array of effects applied onto image
}
