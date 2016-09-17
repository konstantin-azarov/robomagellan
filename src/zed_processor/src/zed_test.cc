///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

//standard includes
#include <stdio.h>
#include <string.h>
#include <ctime>
#include <chrono>

//opencv includes
#include <opencv2/opencv.hpp>

//ZED Includes
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>

//main  function
int main(int argc, char **argv) {
   
    sl::zed::Camera* zed;
	zed = new sl::zed::Camera(sl::zed::HD720);

    // define a struct of parameters for the initialization
    sl::zed::InitParams params;

    //activate verbosity in console window. 
    params.verbose=true;
    params.mode = sl::zed::NONE;
    params.disableSelfCalib = true;
 
    sl::zed::ERRCODE err = zed->init(params);
    std::cout << "Error code : " << sl::zed::errcode2str(err) << std::endl;
    if (err != sl::zed::SUCCESS) {// Quit if an error occurred
        delete zed;
        return 1;
    }

    int width = zed->getImageSize().width;
    int height = zed->getImageSize().height;

    /* Init mouse callback */
    sl::zed::Mat depth;
	zed->grab(sl::zed::STANDARD);
    depth = zed->retrieveMeasure(sl::zed::MEASURE::DEPTH); // Get the pointer
   
    //create Opencv Windows
    cv::namedWindow("VIEW", cv::WINDOW_AUTOSIZE);

    //Jetson only. Execute the calling thread on core 2
    sl::zed::Camera::sticktoCPUCore(2);

    //loop until 'q' is pressed
	cv::Mat left(cv::Size(width, height), CV_8UC3);
	char key = 'a';
    while (key != 'q') 
	{
		auto start = std::chrono::high_resolution_clock::now(); //start the chrono
		if (!zed->grab(sl::zed::STANDARD, false, false, false)) {
			auto grabDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
			std::cout << "grab: " << grabDuration.count() << "ms    ";
			left = sl::zed::slMat2cvMat(zed->retrieveImage(sl::zed::LEFT_UNRECTIFIED));
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
			std::cout << "grab&retrieveImage: " << duration.count() << "ms" << std::endl;
			cv::imshow("VIEW", left);
			key = cv::waitKey(5);
		}
    }

    delete zed;
    return 0;
}
