/*********************************************************************************************************************
* MCX Vision Opensourec Library 即（MCX Vision 开源库）是一个基于官方 SDK 接口的第三方开源库
* Copyright (c) 2024 SEEKFREE 逐飞科技
* 
* 本文件是 MCX Vision 开源库的一部分
* 
* MCX Vision 开源库 是免费软件
* 您可以根据自由软件基金会发布的 GPL（GNU General Public License，即 GNU通用公共许可证）的条款
* 即 GPL 的第3版（即 GPL3.0）或（您选择的）任何后来的版本，重新发布和/或修改它
* 
* 本开源库的发布是希望它能发挥作用，但并未对其作任何的保证
* 甚至没有隐含的适销性或适合特定用途的保证
* 更多细节请参见 GPL
* 
* 您应该在收到本开源库的同时收到一份 GPL 的副本
* 如果没有，请参阅<https://www.gnu.org/licenses/>
* 
* 额外注明：
* 本开源库使用 GPL3.0 开源许可证协议 以上许可申明为译文版本
* 许可申明英文版在 libraries/doc 文件夹下的 GPL3_permission_statement.txt 文件中
* 欢迎各位使用并传播本程序 但修改内容时必须保留逐飞科技的版权声明（即本声明）
* 
* 文件名称          zf_model_process
* 公司名称          成都逐飞科技有限公司
* 版本信息          查看 libraries/doc 文件夹内 version 文件 版本说明
* 开发环境          MDK 5.38a
* 适用平台          MCX Vision
* 店铺链接          https://seekfree.taobao.com/
* 
* 修改记录
* 日期              作者                备注
* 2024-04-21        ZSY            first version
********************************************************************************************************************/
#include "zf_model_process.h"

#include "model.h"
#include "yolo_post_processing.h"
#include "zf_device_ips200.h"

#define MODEL_IN_W	                (160 )
#define MODEL_IN_H                  (128 )
#define MODEL_IN_C	                (3   )
#define MAX_OD_BOX_CNT              (12  )
#define SWAPBYTE(h)                 ((((uint16_t)h << 8)&0xFF00) | ((uint16_t)h >> 8))
//画面大小320 240
typedef struct tagodresult_t
{
    union
    {
        int16_t xyxy[4];
        struct
        {
            int16_t x1;
            int16_t y1;
            int16_t x2;
            int16_t y2;
        };
    };
    float score;
    int label;
} odresult_t;



tensor_dims_t inputDims;
tensor_type_t inputType;
uint8_t* inputData;
tensor_dims_t outputDims;
tensor_type_t outputType;
uint8_t* outputData;
size_t arenaSize;
uint32_t out_size;
yolo::object_detection::PostProcessParams postProcessParams;
TfLiteTensor* outputTensor[3];
float* anchors;
std::vector<yolo::object_detection::DetectionResult> results;

int s_odretcnt = 0;
__attribute__((section(".model_input_buffer"))) volatile uint8_t __attribute((aligned(8))) model_input_buf[MODEL_IN_W * MODEL_IN_H * MODEL_IN_C] = {0};
odresult_t s_odrets[MAX_OD_BOX_CNT];

void draw_rect_on_slice_buffer(uint16_t* pcam, int srcw, int cury, int stride, odresult_t* podret, int retcnt, int slice_height)
{
    int i = 0;
    for(i = 0; i < retcnt; i++, podret++)
    {
        uint32_t color = 0xffffffff;
        uint16_t* phorline = 0;
        int stripey1 = podret->y1 - cury;
        if(stripey1 >= 0 && stripey1 < slice_height)
        {
            for(int j = 0; j < 2 && stripey1 + j < slice_height; j++)
            {
                phorline = pcam + srcw * (j + stripey1) + podret->x1;
                memset(phorline, color, (podret->x2 - podret->x1) * 2);
            }
        }

        int stripey2 = podret->y2 - cury;
        if(stripey2 >= 0 && stripey2 < slice_height)
        {
            for(int j = 0; j < 2 && stripey2 + j < slice_height; j++)
            {
                phorline = pcam + srcw * (j + stripey2) + podret->x1;
                memset(phorline, color, (podret->x2 - podret->x1) * 2);
            }
        }

        uint16_t* pvtclinel = pcam + podret->x1;
        uint16_t* pvtcliner = pcam + podret->x2 - 2;

        for(int y = cury; y < cury + slice_height; y++)
        {
            if(y > podret->y1 && y < podret->y2)
            {
                memset(pvtclinel, color, 4);
                memset(pvtcliner, color, 4);
            }
            pvtclinel += srcw;
            pvtcliner += srcw;
        }

    }
}

void rgb565stridedtorgb888(const uint16_t* pin, int srcw, int wndw, int wndh, int wndx0, int wndy0,
                           uint8_t* p888, int stride, uint8_t issub128)
{
    const uint16_t* psrc = pin;
    uint32_t datin, datout, datouts[3];
    uint8_t* p888out = p888;

    for(int y = wndy0, y1 = (wndh - wndy0) / stride - wndy0; y < wndh; y += stride, y1--)
    {
        psrc = pin + srcw * y + wndx0;
        for(int x = 0; x < wndw; x += stride * 4)
        {
            datin = SWAPBYTE(psrc[0]);
            psrc += stride;
            datouts[0] = (datin & 31) << 19 | (datin & 63 << 5) << 5 | ((datin >> 8) & 0xf8);

            datin = SWAPBYTE(psrc[0]);
            psrc += stride;
            datout = (datin & 31) << 19 | (datin & 63 << 5) << 5 | ((datin >> 8) & 0xf8);
            datouts[0] |= datout << 24;
            datouts[1] = datout >> 8;

            datin = SWAPBYTE(psrc[0]);
            psrc += stride;
            datout = (datin & 31) << 19 | (datin & 63 << 5) << 5 | ((datin >> 8) & 0xf8);
            datouts[1] |= (datout << 16) & 0xffff0000;
            datouts[2] = datout >> 16;

            datin = SWAPBYTE(psrc[0]);
            psrc += stride;
            datout = (datin & 31) << 19 | (datin & 63 << 5) << 5 | ((datin >> 8) & 0xf8);

            datouts[2] |= datout << 8;

            if(issub128)
            {
                datouts[0] ^= 0x80808080;
                datouts[1] ^= 0x80808080;
                datouts[2] ^= 0x80808080;
            }
            memcpy(p888out, datouts, 3 * 4);
            p888out += 3 * 4;
        }
    }
}

void ezh_copy_slice_to_model_input(uint32_t idx, uint32_t cam_slice_buffer, uint32_t cam_slice_width, uint32_t cam_slice_height, uint32_t max_idx)
{
    static uint8_t* pCurDat;
    uint32_t curY;
    uint32_t s_imgStride = cam_slice_width / MODEL_IN_W;

    if(idx > max_idx)
    {
        return;
    }
    uint32_t ndx = idx;
    curY = ndx * cam_slice_height;
    int wndY = (s_imgStride - (curY - 0) % s_imgStride) % s_imgStride;

    pCurDat = (uint8*)model_input_buf + 3 * ((cam_slice_height * ndx + wndY) * cam_slice_width / s_imgStride / s_imgStride);

    if(curY + cam_slice_height >= 0)
    {
        rgb565stridedtorgb888((uint16_t*)cam_slice_buffer, cam_slice_width, cam_slice_width, cam_slice_height, 0, wndY, pCurDat, s_imgStride, 1);
    }
}
typedef struct
{
    uint16 res_x1;
    uint16 res_y1;
    uint16 res_x2;
    uint16 res_y2;
}od_result_t;
volatile od_result_t od_result;

typedef struct {
    double x;   // 坐标
    double P;   // 估计误差协方差
    double Q;   // 过程噪声协方差
    double R;   // 测量噪声协方差
    double K;   // 卡尔曼增益
} KalmanFilter;

// 初始化卡尔曼滤波器
void kalman_init(KalmanFilter* kf, double Q, double R) {
    kf->x = 0.0;
    kf->P = 1.0;
    kf->Q = Q;
    kf->R = R;
    kf->K = 0.0;
}

// 卡尔曼滤波器更新
double kalman_update(KalmanFilter* kf, double measurement) {
    // 预测
    kf->P += kf->Q;

    // 计算卡尔曼增益
    kf->K = kf->P / (kf->P + kf->R);

    // 更新估计值
    kf->x += kf->K * (measurement - kf->x);

    // 更新误差协方差
    kf->P *= (1 - kf->K);

    return kf->x;
}

KalmanFilter kf_x, kf_y;
//-------------------------------------------------------------------------------------------------------------------
// 函数简介     模型的目标信息输出
// 参数说明     odres           目标结构体
// 参数说明     retcnt          目标数量
// 返回参数     void
// 使用示例     zf_model_odresult_out(s_odrets, s_odretcnt);
// 备注信息     
//-------------------------------------------------------------------------------------------------------------------
int16 target_point[2] = {140,150};  																							
//设定的距离，小于此距离认为卡片已经靠近小车
int16 set_distance = 200;       
int16 set_distance_x = 200;       

//计算卡片的距离
int16 distance = 0;                																							 
//矫正完成标志，1为检测到卡片
uint8 correct_flag = 0;          																							   
//误识别计数
uint8 mistake_dection_cnt = 0;	 																
//reset:0
//locate:1
//detect:2
//num+abc:3
//put:4
int mode=1;
uint8 u5_array[7]={'@',0,0,0,0,0,'#'};
void u5_tx(uint8 center_x,uint8 center_y,uint8 distance, uint8 correctflag, uint8 mode)
{
	u5_array[1]=center_x;
	u5_array[2]=center_y;
	u5_array[3]=distance;
	u5_array[4]=correctflag;
	u5_array[5]=mode;
	user_uart_write_buffer (u5_array, 7);
}

void zf_model_odresult_out(const odresult_t* odres, int retcnt)
{
uint16_t center_x = 0, center_y = 0;
		int16 shortest;
	uint8 area=0;
		if(mode==1)
			shortest = set_distance;
		if(mode==3)
			shortest = set_distance_x;
		
		//清零上次标志
    correct_flag = 0;   																														
		//清零上次计数
		mistake_dection_cnt = 0;																												
		//没找到卡片
    if(retcnt == 0) 																																
    {																																										
				//就将初始化坐标直接输出
//				zf_debug_printf("@%d,%d,%d,%d#\n", center_x, center_y, distance, correct_flag);
				u5_tx(center_x, center_y, distance, correct_flag, mode);
			  return;
    }
		//遍历每幅图片中检测到的卡片,找到距离最近的卡片
    for(int i = 0; i < retcnt; i++, odres++)    																	
    {
//        zf_debug_printf("sc:%d,x1:%d,y1:%d,x2:%d,y2:%d\r\n",
//                        (int)(odres->score*100), odres->x1, odres->y1, odres->x2, odres->y2);
        
        // 通过用户串口-串口5发送数据，通过其他单片机接收
        // od_result.res_x1 = odres->x1;
        // od_result.res_y1 = odres->y1;
        // od_result.res_x2 = odres->x2;
        // od_result.res_y2 = odres->y2;
        // user_uart_putchar(0xAA);
        // user_uart_putchar((uint8)i);
        // user_uart_write_buffer((const uint8*)(&od_result), sizeof(od_result));
        // user_uart_putchar(0xFF);
		
				//求中心点坐标
				int16_t temp_x = (odres->x1 + odres->x2)/2;																		
				int16_t temp_y = (odres->y1 + odres->y2)/2;																		
				//求长宽
				int16_t temp_w = (odres->x2 - odres->x1);																					
				int16_t temp_h = (odres->y2 - odres->y1);
				//过滤掉一些误识别的情况（有时赛道会误识别）
				//误识别情况：
				//1.中心坐标y值靠画面下方，
				//2.画面面积过小，
				//3.宽大于高的三倍不符合正方形
				//4.距离底线中点过远
				//5.画面面积过大
				if(mode == 1)
				{
					distance = sqrt( pow(temp_x - target_point[0], 2) + pow(temp_y - target_point[1], 2) );
					//ips200_show_int(temp_x, temp_y-20, temp_w*temp_h);

					if(temp_y>SCC8660_H-15 || temp_w*temp_h<3000 || temp_w>2*temp_h ||temp_h>2*temp_w || distance>set_distance || temp_w*temp_h>15000)
					{
						mistake_dection_cnt++;
						continue;
					}
					//更新最近距离的卡片
					else if(distance<shortest)       
					{
						shortest = distance;
						
						center_x = (odres->x1 + odres->x2)/2;
						center_y = (odres->y1 + odres->y2)/2;
					}
				}
				if(mode == 3)
				{
					distance = sqrt(pow(temp_x - target_point[0], 2));
					if(temp_y>SCC8660_H-15 || temp_w*temp_h<3000 || temp_w>3*temp_h || distance>set_distance_x || temp_w*temp_h>15000)
					{
						mistake_dection_cnt++;
						continue;
					}
					//更新最近距离的卡片
					if(distance<shortest)       
					{
						shortest = distance;
						center_x = (odres->x1 + odres->x2)/2;
						center_y = (odres->y1 + odres->y2)/2;
						area=temp_w*temp_h;
					}
				}
    }
		//全是为误识别
		if(mistake_dection_cnt == retcnt)	
		{
			center_x = 0;
			center_y = 0;
			correct_flag = 0;
//			zf_debug_printf("@%d,%d,%d,%d#\n", center_x, center_y, distance, correct_flag);
			u5_tx(center_x, center_y, distance, correct_flag, mode);
			return;
		}
		//不为零，说明上面的if判断执行过，并且错误检测数量小于总数量
		//说明最终最近距离记录的坐标为有效结果
    if(center_x != 0 && center_y != 0 && mistake_dection_cnt<retcnt) 
    {  
				//打印中心坐标，还有离点（160，170）的距离
				ips200_show_int(center_x, center_y, center_x);
				ips200_show_int(center_x+30, center_y, center_y);
			  ips200_show_int(center_x, center_y+20 ,mode);
				ips200_show_int(center_x+30, center_y+20, distance);
        correct_flag = 1;
//				zf_debug_printf("@%d,%d,%d,%d#\n", center_x, center_y, distance, correct_flag);
        //串口发送坐标，距离，是否检测到
				u5_tx(center_x, center_y, distance, correct_flag, mode);
				zf_debug_printf("@%d,%d,%d,%d,%d#\n", center_x, center_y, distance, correct_flag, mode);
				zf_debug_printf("mianji:%d",area);


        //在ips上面画框
				draw_rect_on_slice_buffer((scc8660_image), SCC8660_W,  0, 1, s_odrets, s_odretcnt, SCC8660_H);
				ips200_show_scc8660(scc8660_image);
    }
}

//-------------------------------------------------------------------------------------------------------------------
// 函数简介     模型初始化
// 返回参数     void
// 使用示例     zf_model_init();
// 备注信息     
//-------------------------------------------------------------------------------------------------------------------
void zf_model_init(void)
{
if(MODEL_Init() != kStatus_Success)
    {
        zf_debug_printf("Failed initializing model");
				//如果状态不成功就一直循环？？
        while(1);
    }
		//当前已用的字节数=MODEL_GetArenaUsedBytes(总的可用字节数）
    size_t usedSize = MODEL_GetArenaUsedBytes(&arenaSize);
		//打印已经使用的张量区的状态，将字节数转换为千字节（kB）并计算使用百分比
    zf_debug_printf("\r\n%d/%d kB (%0.2f%%) tensor arena used\r\n", usedSize / 1024, arenaSize / 1024, 100.0 * usedSize / arenaSize);
		//MODEL_GetInputTensorData() 函数获取模型的输入张量的数据和维度
		//这些信息存储在 inputDims 和 inputType 中。
		//inputData 存储输入张量的数据指针。
    inputData = MODEL_GetInputTensorData(&inputDims, &inputType);
		//获取输出张量的数据和维度，并将其存储在 outputData、outputDims 和 outputType 中。
    outputData = MODEL_GetOutputTensorData(&outputDims, &outputType);
    //调用 MODEL_GetOutputSize() 函数获取输出张量的大小，并将其存储到 out_size 中。
		out_size = MODEL_GetOutputSize();
		//从 inputDims 中提取输入图像的行数和列数，并将其存储在 postProcessParams 结构的相应字段中。
    postProcessParams.inputImgRows = (int)inputDims.data[1];
    postProcessParams.inputImgCols =	(int)inputDims.data[2];
		//将输出大小存储到 postProcessParams.output_size 中。
    postProcessParams.output_size = (int)out_size;
    postProcessParams.originalImageWidth = SCC8660_W;
    postProcessParams.originalImageHeight = SCC8660_H;
		//设置一些后处理参数：
		//threshold: 检测阈值，值为 0.4。
		//nms: 非极大值抑制阈值，值为 0.2。
		//numClasses: 类别数目，这里设为 1。
		//topN: 可选的选项，设为 0。
    postProcessParams.threshold = 0.45;	//0.4
    postProcessParams.nms = 0.2;
    postProcessParams.numClasses = 1;
    postProcessParams.topN = 0;
    //获取模型的锚框（anchors），可能用于目标检测算法。
    anchors = MODEL_GetAnchors();
		//通过一个循环遍历每个输出索引，调用 MODEL_GetOutputTensor(i) 来获取每个输出张量，并将其存储在 outputTensor[i] 中。
    for(int i = 0; i < out_size; i ++)
    {
        outputTensor[i] = MODEL_GetOutputTensor(i);
        postProcessParams.anchors[i][0] = *(anchors + 6 * (out_size - 1 - i));
        postProcessParams.anchors[i][1] = *(anchors + 6 * (out_size - 1 - i) + 1);
        postProcessParams.anchors[i][2] = *(anchors + 6 * (out_size - 1 - i) + 2);
        postProcessParams.anchors[i][3] = *(anchors + 6 * (out_size - 1 - i) + 3);
        postProcessParams.anchors[i][4] = *(anchors + 6 * (out_size - 1 - i) + 4);
        postProcessParams.anchors[i][5] = *(anchors + 6 * (out_size - 1 - i) + 5);
    }
}


//-------------------------------------------------------------------------------------------------------------------
// 函数简介     模型运行
// 返回参数     void
// 使用示例     zf_model_run();
// 备注信息     
//-------------------------------------------------------------------------------------------------------------------
uint8 buffer[6] = {0};
uint8_t count_cards;
float dist2car;
typedef struct card{
	uint8_t x;
	uint8_t y;
	float score;
}detected_card;
detected_card d_cards[4];


#define ALPHA 0.1
void low_pass_filter(double x, double y, double *x_filtered, double *y_filtered) {
    static double x_prev = 0.0;
    static double y_prev = 0.0;
    static int first_run = 1;

    // 初始情况下，滤波后的值设为输入值
    if (first_run) {
        *x_filtered = x;
        *y_filtered = y;
        x_prev = x;
        y_prev = y;
        first_run = 0;
    } else {
        // 低通滤波公式
        *x_filtered = ALPHA * x + (1 - ALPHA) * x_prev;
        *y_filtered = ALPHA * y + (1 - ALPHA) * y_prev;

        // 更新前一次的值
        x_prev = *x_filtered;
        y_prev = *y_filtered;
    }
}

void zf_model_run(void)
{
 uint8_t* buf = 0;
    memset(inputData, 0, inputDims.data[1]*inputDims.data[2]*inputDims.data[3]);
    buf = inputData + (inputDims.data[1] - MODEL_IN_H) / 2 * MODEL_IN_W * MODEL_IN_C;
    memcpy(buf, (uint8*)model_input_buf, MODEL_IN_W * MODEL_IN_H * MODEL_IN_C);

    results.clear();
    MODEL_RunInference();

    s_odretcnt = 0;
    if(!(yolo::DetectorPostProcess((const TfLiteTensor**)outputTensor,results, postProcessParams).DoPostProcess()))
    {
        s_odretcnt = 0;
    }
		//开始一个范围-based 循环，遍历 results 中的每个结果。
    for(const auto& result : results)
    {
				//检查结果的标准化值（m_normalisedVal）是否高于设定的阈值（postProcessParams.threshold）。
        if(result.m_normalisedVal > postProcessParams.threshold)
        {
					//如果满足条件，将结果的边界框坐标（左上角和右下角）以及置信度分数存储在 s_odrets 数组中。
					//每次存储后，计数器 s_odretcnt 加一。
            s_odrets[s_odretcnt].x1 = result.m_x0;
            s_odrets[s_odretcnt].x2 = result.m_x0 + result.m_w;
            s_odrets[s_odretcnt].y1 = result.m_y0;
            s_odrets[s_odretcnt].y2 = result.m_y0 + result.m_h;
            s_odrets[s_odretcnt].score = result.m_normalisedVal;
            s_odretcnt ++;
        }
    }
		//若定义了 IS_UART_OUTPUT_ODRESULT，则向 UART 输出对象检测结果。
#if IS_UART_OUTPUT_ODRESULT
    if(1)//if(s_odretcnt > 0)
    {
        zf_model_odresult_out(s_odrets, s_odretcnt);
    }
#endif
#if IS_SHOW_SCC8660
		//如果 s_odretcnt 大于 0，则在图像 scc8660_image 上绘制检测到的边界框。
    if(s_odretcnt)
    {
        draw_rect_on_slice_buffer((scc8660_image), SCC8660_W,  0, 1, s_odrets, s_odretcnt, SCC8660_H);
    }
		ips200_draw_point (target_point[0], target_point[1],  RGB565_RED);
    ips200_show_scc8660(scc8660_image);
#endif
		

}

/*

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>  // 确保使用正确的类型

#define THRESHOLD 50        // 跳变阈值
#define STABILITY_FRAMES 10  // 稳定性帧数

typedef struct {
    int32_t x;
    int32_t y;
    int jump_count;
} FilterState;

void initialize_filter_state(FilterState* state, int32_t x, int32_t y) {
    state->x = x;
    state->y = y;
    state->jump_count = 0;
}

void detect_and_filter_jumps(FilterState* state, int32_t x0, int32_t y0, int32_t threshold, int stability_frames, int32_t* x_filtered, int32_t* y_filtered) {
    int32_t dx = abs(x0 - state->x);
    int32_t dy = abs(y0 - state->y);

    if (dx < threshold && dy < threshold) {
        state->x = x0;
        state->y = y0;
        state->jump_count = 0;  // 重置跳变计数器
    } else {
        state->jump_count++;
        if (state->jump_count >= stability_frames) {
            state->x = x0;  // 认为这是新的目标
            state->y = y0;
            state->jump_count = 0;  // 重置跳变计数器
        }
    }

    *x_filtered = state->x;
    *y_filtered = state->y;
}

void zf_model_run(void)
{
    uint8 buffer[5] = {0};
    // 通信格式："@{conf}{x}{y}$"
    buffer[0] = (uint8)0X0A;
    buffer[4] = (uint8)0X0D;
    // ********************************************************************************
    uint8_t* buf = 0;
    memset(inputData, 0, inputDims.data[1]*inputDims.data[2]*inputDims.data[3]);
    buf = inputData + (inputDims.data[1] - MODEL_IN_H) / 2 * MODEL_IN_W * MODEL_IN_C;
    memcpy(buf, (uint8*)model_input_buf, MODEL_IN_W * MODEL_IN_H * MODEL_IN_C);

    results.clear();
    MODEL_RunInference();

    s_odretcnt = 0;
    if(!(yolo::DetectorPostProcess((const TfLiteTensor**)outputTensor,results, postProcessParams).DoPostProcess()))
    {
        s_odretcnt = 0;
    }

    static FilterState filter_state = {0, 0, 0};
    static int filter_initialized = 0;

    for(const auto& result : results)
    {
        if(result.m_normalisedVal > postProcessParams.threshold)
        {
            if (!filter_initialized) {
                initialize_filter_state(&filter_state, (int32_t)result.m_x0, (int32_t)result.m_y0);
                filter_initialized = 1;
            }

            int32_t x_filtered, y_filtered;
            detect_and_filter_jumps(&filter_state, (int32_t)result.m_x0, (int32_t)result.m_y0, THRESHOLD, STABILITY_FRAMES, &x_filtered, &y_filtered);

            s_odrets[s_odretcnt].x1 = x_filtered;
            s_odrets[s_odretcnt].x2 = x_filtered + result.m_w;
            s_odrets[s_odretcnt].y1 = y_filtered;
            s_odrets[s_odretcnt].y2 = y_filtered + result.m_h;
            s_odrets[s_odretcnt].score = result.m_normalisedVal;
            // ***********************************************************************
            if (((y_filtered > 120) && (start_flag == 1))
            || (locate_flag == 1 && y_filtered >= 100)						
            || (goto_flag == 1 && result.m_normalisedVal >= 0.6)) {
                buffer[1] = 0x01;
                buffer[2] = (uint8_t)x_filtered;
                buffer[3] = (uint8_t)y_filtered;
            }
            s_odretcnt ++;
        }
    }
    // *****
    switch (mode) {
        case 0:ips200_show_string(180,10,"RESET");break;
        case 1:ips200_show_string(180,10,"Detecting");break;
        case 2:ips200_show_string(180,10,"Locating");break;
        default:ips200_show_string(180,10,"default");break;
    }
    user_uart_write_buffer(buffer, sizeof(buffer));
    // *****
#if IS_UART_OUTPUT_ODRESULT
    if(s_odretcnt > 0)
    {
        zf_model_odresult_out(s_odrets, s_odretcnt);
    }
#endif
#if IS_SHOW_SCC8660
    if(s_odretcnt)
    {
        draw_rect_on_slice_buffer((scc8660_image), SCC8660_W,  0, 1, s_odrets, s_odretcnt, SCC8660_H);
    }
    ips200_show_scc8660(scc8660_image);
#endif
}
*/
