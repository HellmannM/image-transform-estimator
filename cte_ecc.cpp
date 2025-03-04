#include "cte_ecc.h"

extern "C" {
    image_transform_estimator* create_estimator()
    {
        return dynamic_cast<image_transform_estimator*>(new ecc());
    }

    void destroy_estimator(image_transform_estimator* estimator)
    {
        delete dynamic_cast<ecc*>(estimator);
    }

    const char* get_estimator_type()
    {
        return "ECC";
    }

    const char* get_estimator_description()
    {
        return "Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization";
    }
}
