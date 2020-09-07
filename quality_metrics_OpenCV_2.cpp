#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>

using namespace std;
using namespace cv;

// quality-metric
namespace qm
{
#define C1 (float)(0.01 * 100 * 0.01 * 100)
#define C2 (float)(0.03 * 100 * 0.03 * 100)

    // sigma on block_size
    double sigma(Mat &m, int i, int j, int block_size)
    {
        double sd = 0;

        Mat m_tmp = m(Range(i, i + block_size), Range(j, j + block_size));
        Mat m_squared(block_size, block_size, CV_64F);

        multiply(m_tmp, m_tmp, m_squared);

        // E(x)
        double avg = mean(m_tmp)[0];
        // E(x²)
        double avg_2 = mean(m_squared)[0];

        sd = sqrt(avg_2 - avg * avg);

        return sd;
    }

    // Covariance
    double cov(Mat &m1, Mat &m2, int i, int j, int block_size)
    {
        Mat m3 = Mat::zeros(block_size, block_size, m1.depth());
        Mat m1_tmp = m1(Range(i, i + block_size), Range(j, j + block_size));
        Mat m2_tmp = m2(Range(i, i + block_size), Range(j, j + block_size));

        multiply(m1_tmp, m2_tmp, m3);

        double avg_ro = mean(m3)[0];    // E(XY)
        double avg_r = mean(m1_tmp)[0]; // E(X)
        double avg_o = mean(m2_tmp)[0]; // E(Y)

        double sd_ro = avg_ro - avg_o * avg_r; // E(XY) - E(X)E(Y)

        return sd_ro;
    }

    // Mean squared error
    double eqm(Mat &img1, Mat &img2)
    {
        int i, j;
        double eqm = 0;
        int height = img1.rows;
        int width = img1.cols;
        int cnt = 0;

        for (i = 0; i < height; i++)
        {
            for (j = 0; j < width; j++)
            {
                if (img1.at<double>(i, j) > 0 && img2.at<double>(i, j) > 0)
                {
                    eqm += (img1.at<double>(i, j) - img2.at<double>(i, j)) * (img1.at<double>(i, j) - img2.at<double>(i, j));
                    cnt++;
                }
            }
        }

        if (cnt == 0)
        {
            return 1e9;
        }
        else
        {
            return eqm / cnt;
        }
    }

    /**
	 *	Compute the PSNR between 2 images
	 */
    double psnr(Mat &img_src, Mat &img_compressed, int block_size)
    {
        int D = 255;
        return (10 * log10((D * D) / eqm(img_src, img_compressed)));
    }

    /**
	 * Compute the SSIM between 2 images
	 */
    double ssim(Mat &img1, Mat &img2, int block_size, bool show_progress = false)
    {
        double mssim = 0;

        int nbBlockPerHeight = img1.rows / block_size;
        int nbBlockPerWidth = img1.cols / block_size;
        int validBlocks = 0;

        for (int k = 0; k < nbBlockPerHeight; k++)
        {
            for (int l = 0; l < nbBlockPerWidth; l++)
            {
                int m = k * block_size;
                int n = l * block_size;

                int cnt = 0;
                double avg_o;
                double avg_r;
                double avg2_o;
                double avg2_r;
                double avg_or;
                for (int i = 0; i < block_size; i++)
                {
                    for (int j = 0; j < block_size; j++)
                    {
                        double o = img1.at<double>(m + i, n + j);
                        double r = img2.at<double>(m + i, n + j);
                        if (o > 0 && r > 0)
                        {
                            avg_o += o;
                            avg2_o += o * o;
                            avg_r += r;
                            avg2_r += r * r;
                            avg_or += o * r;
                            cnt++;
                        }
                    }
                }

                if (cnt == 0)
                {
                }
                else
                {
                    avg_o /= cnt;
                    avg2_o /= cnt;
                    avg_r /= cnt;
                    avg2_r /= cnt;
                    avg_or /= cnt;

                    double sigma2_o = avg2_o - avg_o * avg_o;
                    double sigma2_r = avg2_r - avg_r * avg_r;
                    double sigma_or = avg_or - avg_o * avg_r;

                    double ssim = ((2 * avg_o * avg_r + C1) * (2 * sigma_or + C2)) / ((avg_o * avg_o + avg_r * avg_r + C1) * (sigma2_o + sigma2_r + C2));
                    ssim = min(1.0, ssim);
                    ssim = max(0.0, ssim);
                    mssim += ssim;

                    validBlocks++;
                }
            }
        }

        if (validBlocks == 0)
        {
            return 0;
        }
        else
        {
            return mssim / validBlocks;
        }
    }

    void compute_quality_metrics(char *file1, char *file2, int block_size)
    {

        Mat img_src;
        Mat img_compressed;

        // Loading pictures
        img_src = imread(file1, CV_LOAD_IMAGE_GRAYSCALE);
        img_compressed = imread(file2, CV_LOAD_IMAGE_GRAYSCALE);

        img_src.convertTo(img_src, CV_64F);
        img_compressed.convertTo(img_compressed, CV_64F);

        int height_o = img_src.rows;
        int height_r = img_compressed.rows;
        int width_o = img_src.cols;
        int width_r = img_compressed.cols;

        // Check pictures size
        if (height_o != height_r || width_o != width_r)
        {
            cout << "Images must have the same dimensions" << endl;
            return;
        }

        // Check if the block size is a multiple of height / width
        if (height_o % block_size != 0 || width_o % block_size != 0)
        {
            cout << "WARNING : Image WIDTH and HEIGHT should be divisible by BLOCK_SIZE for the maximum accuracy" << endl
                 << "HEIGHT : " << height_o << endl
                 << "WIDTH : " << width_o << endl
                 << "BLOCK_SIZE : " << block_size << endl
                 << endl;
        }

        double ssim_val = ssim(img_src, img_compressed, block_size);
        double psnr_val = psnr(img_src, img_compressed, block_size);

        cout << "SSIM : " << ssim_val << endl;
        cout << "PSNR : " << psnr_val << endl;
    }
} // namespace qm