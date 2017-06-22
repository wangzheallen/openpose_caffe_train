#ifndef CAFFE_OPENPOSE_CPM_DATA_TRANSFORMER_HPP
#define CAFFE_OPENPOSE_CPM_DATA_TRANSFORMER_HPP

#include <vector>
// OpenPose: added
#ifdef USE_OPENCV
    #include <opencv2/core/core.hpp> // cv::Mat, cv::Point, cv::Size
#endif  // USE_OPENCV
// OpenPose: added end
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class CPMDataTransformer {
public:
    explicit CPMDataTransformer(const CPMTransformationParameter& param, Phase phase);
    virtual ~CPMDataTransformer() {}

    /**
     * @brief Initialize the Random number generations if needed by the
     *    transformation.
     */
    void InitRand();

    /**
     * @brief Applies the transformation defined in the data layer's
     * transform_param block to the data.
     *
     * @param datum
     *    Datum containing the data to be transformed.
     * @param transformed_blob
     *    This is destination blob. It can be part of top blob's data if
     *    set_cpu_data() is used. See data_layer.cpp for an example.
     */
    void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);

    /**
     * @brief Applies the transformation defined in the data layer's
     * transform_param block to a vector of Datum.
     *
     * @param datum_vector
     *    A vector of Datum containing the data to be transformed.
     * @param transformed_blob
     *    This is destination blob. It can be part of top blob's data if
     *    set_cpu_data() is used. See memory_layer.cpp for an example.
     */
    void Transform(const vector<Datum> & datum_vector,
                   Blob<Dtype>* transformed_blob);

#ifdef USE_OPENCV
    /**
     * @brief Applies the transformation defined in the data layer's
     * transform_param block to a vector of Mat.
     *
     * @param mat_vector
     *    A vector of Mat containing the data to be transformed.
     * @param transformed_blob
     *    This is destination blob. It can be part of top blob's data if
     *    set_cpu_data() is used. See memory_layer.cpp for an example.
     */
    void Transform(const vector<cv::Mat> & mat_vector,
                   Blob<Dtype>* transformed_blob);

    /**
     * @brief Applies the transformation defined in the data layer's
     * transform_param block to a cv::Mat
     *
     * @param cv_img
     *    cv::Mat containing the data to be transformed.
     * @param transformed_blob
     *    This is destination blob. It can be part of top blob's data if
     *    set_cpu_data() is used. See image_data_layer.cpp for an example.
     */
    void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob);
#endif  // USE_OPENCV

    /**
     * @brief Applies the same transformation defined in the data layer's
     * transform_param block to all the num images in a input_blob.
     *
     * @param input_blob
     *    A Blob containing the data to be transformed. It applies the same
     *    transformation to all the num images in the blob.
     * @param transformed_blob
     *    This is destination blob, it will contain as many images as the
     *    input blob. It can be part of top blob's data.
     */
    void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);

    /**
     * @brief Infers the shape of transformed_blob will have when
     *    the transformation is applied to the data.
     *
     * @param datum
     *    Datum containing the data to be transformed.
     */
    vector<int> InferBlobShape(const Datum& datum);
    /**
     * @brief Infers the shape of transformed_blob will have when
     *    the transformation is applied to the data.
     *    It uses the first element to infer the shape of the blob.
     *
     * @param datum_vector
     *    A vector of Datum containing the data to be transformed.
     */
    vector<int> InferBlobShape(const vector<Datum> & datum_vector);
    /**
     * @brief Infers the shape of transformed_blob will have when
     *    the transformation is applied to the data.
     *    It uses the first element to infer the shape of the blob.
     *
     * @param mat_vector
     *    A vector of Mat containing the data to be transformed.
     */
#ifdef USE_OPENCV
    vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);
    /**
     * @brief Infers the shape of transformed_blob will have when
     *    the transformation is applied to the data.
     *
     * @param cv_img
     *    cv::Mat containing the data to be transformed.
     */
    vector<int> InferBlobShape(const cv::Mat& cv_img);
#endif  // USE_OPENCV

protected:
     /**
     * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
     *
     * @param n
     *    The upperbound (exclusive) value of the random number.
     * @return
     *    A uniformly random integer value from ({0, 1, ..., n-1}).
     */
    virtual int Rand(int n);

    void Transform(const Datum& datum, Dtype* transformedData);
    // Tranformation parameters
    // TransformationParameter param_; // OpenPose: commented
    CPMTransformationParameter param_; // OpenPose: added


    shared_ptr<Caffe::RNG> rng_;
    Phase phase_;
    Blob<Dtype> data_mean_;
    vector<Dtype> mean_values_;

    // OpenPose: added
public:
    void Transform_nv(const Datum& datum, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_label_blob, const int counter); //image and label

protected:
    struct AugmentSelection {
        bool flip;
        float degree;
        cv::Size crop;
        float scale;
    };

    struct Joints {
        std::vector<cv::Point2f> joints;
        std::vector<float> isVisible;
    };

    struct MetaData {
        std::string datasetString;
        cv::Size imageSize;
        bool isValidation;
        int numberOtherPeople;
        int peopleIndex;
        int annotationListIndex;
        int writeNumber;
        int totalWriteNumber;
        int epoch;
        cv::Point2f objpos; //objpos_x(float), objpos_y (float)
        float scaleSelf;
        Joints jointsSelf; //(3*16)

        std::vector<cv::Point2f> objPosOthers; //length is numberOtherPeople
        std::vector<float> scaleOthers; //length is numberOtherPeople
        std::vector<Joints> jointsOthers; //length is numberOtherPeople
    };

    int mNumberPartsInLmdb;
    int mNumberParts;
    bool mIsTableSet;
    std::vector<std::vector<float>> mAugmentationDegs;
    std::vector<std::vector<int>> mAugmentationFlips;

    void generateLabelMap(Dtype* transformedLabel, const cv::Mat& image, const MetaData& metaData) const;
    void visualize(const cv::Mat& image, const MetaData& metaData, const AugmentSelection& augmentSelection) const;

    bool augmentationFlip(cv::Mat& imageAugmented, cv::Mat& maskMiss, cv::Mat& maskAll, MetaData& metaData, const cv::Mat& image) const;
    float augmentationRotate(cv::Mat& imageAugmented, cv::Mat& maskMiss, cv::Mat& maskAll, MetaData& metaData, const cv::Mat& imageSource) const;
    float augmentationScale(cv::Mat& imageTemp, cv::Mat& maskMiss, cv::Mat& maskAll, MetaData& metaData, const cv::Mat& image) const;
    cv::Size augmentationCropped(cv::Mat& imageAugmented, cv::Mat& maskMissAugmented, cv::Mat& maskAllAugmented, MetaData& metaData, const cv::Mat& imageTemp,
                                 const cv::Mat& maskMiss, const cv::Mat& maskAll) const;

    void rotatePoint(cv::Point2f& point2f, const cv::Mat& R) const;
    bool onPlane(const cv::Point& point, const cv::Size& imageSize) const;
    void swapLeftRight(Joints& joints) const;
    void setAugmentationTable(const int numData);
    void Transform_nv(Dtype* transformedData, Dtype* transformedLabel, const Datum& datum, const int counter);
    void readMetaData(MetaData& metaData, const string& data, size_t offset3, size_t offset1);
    void transformMetaJoints(MetaData& metaData) const;
    void transformJoints(Joints& joints) const;
    void clahe(cv::Mat& bgrImage, const int tileSize, const int clipLimit) const;
    void putGaussianMaps(Dtype* entry, const cv::Point2f& center, const int stride, const int gridX, const int gridY, const float sigma) const;
    void putVecMaps(Dtype* entryX, Dtype* entryY, cv::Mat& count, const cv::Point2f& centerA, const cv::Point2f& centerB, const int stride,
                    const int gridX, const int gridY, const float sigma, const int thre) const;
    // OpenPose: added end
};

}  // namespace caffe

#endif  // CAFFE_OPENPOSE_CPM_DATA_TRANSFORMER_HPP_
