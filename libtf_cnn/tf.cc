// Author: sumingrui

/*
todo:
1. tf_3dcnn
2. bf + pad
 */

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <vector>

#include "tf.h"
#include "utils.h"
#include "share.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;
using std::vector;

typedef cv::Vec<double, 224> vec224d;
const int lines = 482;
const int samples = 640;
const int bands = 224;

bool ReadRawfile(string rawfilepath, double computeRatio, cv::Mat &img_cube)
{
	FILE *rawfile;
	int calLines = lines * computeRatio;
	img_cube = cv::Mat::zeros(calLines, samples, CV_64FC(224));
	rawfile = fopen(rawfilepath.c_str(), "rb");
	if (rawfile != NULL)
	{
		for (int h = 0; h < calLines; h++)
		{
			fseek(rawfile, h * samples * bands * 2, SEEK_SET);
			uint16_t raw_arr[samples * bands] = {0};
			fread(raw_arr, sizeof(uint16_t), samples * bands, rawfile);
			for (int d = 0; d < bands; d++)
			{
				for (int w = 0; w < samples; w++)
				{
					img_cube.at<vec224d>(h, w)[d] = double(raw_arr[d * samples + w]);
				}
			}
		}
		fclose(rawfile);
		log(info, "load raw file success: " + rawfilepath);
		return true;
	}
	else
	{
		log(warnning, "load raw file failed: " + rawfilepath);
		return false;
	}
}

// Does not configure matlab lib.
/*
bool Draw_gt(string matfilepath, string key)
{
	MATFile *pMatFile = NULL;
	mxArray *pMxArray = NULL;
	pMatFile = matOpen(matfilepath.c_str(), "r");
	pMxArray = matGetVariable(pMatFile, key.c_str());

	const size_t *n = mxGetDimensions(pMxArray);
	int h_gt = *n;
	int w_gt = *(n + 1);

	double *pData = (double *)(mxGetData(pMxArray));
	cv::Mat img_gt = cv::Mat::zeros(h_gt, w_gt, CV_64FC1);
	for (int i = 0; i < w_gt; i++)
	{
		double *source_gt = pData + (i * h_gt);
		for (int j = 0; j < h_gt; j++)
		{
			img_gt.at<double>(j, i) = *(source_gt + j);
		}
	}
	save_imagesc(img_gt, "img_gt.jpg");
	log(info, "Draw ground truth: img_gt.jpg");
}
 */

int TF_2dcnn(const char *c_rawfilepath, const char *c_filename, double computeRatio)
{
	clock_t time_1 = clock();
	string rawfilepath(c_rawfilepath);
	string filename(c_filename);
	// 计算行数
	int calLines = lines * computeRatio;
	cv::Mat img_cube = cv::Mat::zeros(calLines, samples, CV_64FC(224));
	rawfilepath = rawfilepath + filename + ".raw";

	if (!ReadRawfile(rawfilepath, computeRatio, img_cube))
	{
		return 0;
	}

	// norm
	img_cube = norm(img_cube);

	// pad 224->225
	pad_2dcnn(img_cube);

	img_cube.convertTo(img_cube, CV_32F);

	// build seesion and model
	Session *session;
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok())
	{
		log(LOGLEVEL::error, status.ToString());
		return 0;
	}

	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), "/repos/tf_server/AArch64_HSI_Tensorflow_Server/pbfile/2dcnn.pb", &graph_def);
	if (!status.ok())
	{
		log(LOGLEVEL::error, status.ToString());
		return 0;
	}

	status = session->Create(graph_def);
	if (!status.ok())
	{
		log(LOGLEVEL::error, status.ToString());
		return 0;
	}

	// save memory
	int groupnum;
	vector<int> rowArr;
	vector<int> rowStartArr;
	vector<int> rowEndArr;

	if (calLines <= 160)
	{
		groupnum = 1;
		rowArr = {calLines};
		rowStartArr = {0};
		rowEndArr = {calLines};
	}
	else if (calLines <= 320)
	{
		groupnum = 2;
		rowArr = {160, calLines - 160};
		rowStartArr = {0, 160};
		rowEndArr = {160, calLines};
	}
	else
	{
		groupnum = 3;
		rowArr = {160, 160, calLines - 320};
		rowStartArr = {0, 160, 320};
		rowEndArr = {160, 320, calLines};
	}

	cv::Mat img_spectral = cv::Mat::zeros(calLines, samples, CV_64FC1);

	double algorithm_time = 0.0;

	for (int x = 0; x < groupnum; x++)
	{
		vector<std::pair<string, Tensor>> inputs;
		vector<Tensor> outputs;

		typedef cv::Vec<float, 225> vec225f;
		Tensor input_tensor(DT_FLOAT, TensorShape({rowArr[x] * samples, 15, 15, 1}));
		auto input_tensor_mapped = input_tensor.tensor<float, 4>();

		inputs = {{"input_1", input_tensor}};

		for (int nRow = rowStartArr[x]; nRow < rowEndArr[x]; nRow++)
		{
			for (int nCol = 0; nCol < samples; nCol++)
			{
				for (int nChannel = 0; nChannel < 225; nChannel++)
				{
					input_tensor_mapped((nRow - rowStartArr[x]) * samples + nCol, nChannel / 15, nChannel % 15, 0) =
						img_cube.at<vec225f>(nRow, nCol)[nChannel];
				}
			}
		}

		clock_t ai_begin = clock();
		status = session->Run(inputs, {"fc1/Softmax"}, {}, &outputs);
		clock_t ai_end = clock();
		algorithm_time += double(ai_end - ai_begin) / CLOCKS_PER_SEC;

		if (!status.ok())
		{
			log(LOGLEVEL::error, status.ToString());
			return 0;
		}

		auto output = outputs[0].tensor<float, 2>();
		int output_num = outputs[0].shape().dim_size(0);
		int output_dim = outputs[0].shape().dim_size(1);

		for (int n = 0; n < output_num; n++)
		{
			double output_prob = 0.0;
			double class_id = 0.0;
			for (int j = 0; j < output_dim; j++)
			{
				if (output(n, j) >= output_prob)
				{
					class_id = j;
					output_prob = output(n, j);
				}
			}
			img_spectral.at<double>(n / samples + rowStartArr[x], n % samples) = double(class_id);
		}
	}

	clock_t time_2 = clock();
	log(info, "Data processing time: " + std::to_string(double(time_2 - time_1) / CLOCKS_PER_SEC - algorithm_time) + "s");
	log(info, "Algorithm running time: " + std::to_string(algorithm_time) + "s");

	//save opencv mat
	save_xml(img_spectral, "./results/" + filename + "-Xavier.xml");
	save_imagesc(img_spectral, "./results/" + filename + "-Xavier.jpg");
	
	session->Close();

	return 0;
}
