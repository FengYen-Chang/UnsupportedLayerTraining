// 
// for extension layer cosh
// Implement by John Feng
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <utility>
#include <functional>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class coshImpl: public ExtLayerBase {
public:
    explicit coshImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            addConfig(layer, {DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        SizeVector in_dims = inputs[0]->getTensorDesc().getDims();
        SizeVector out_dims = outputs[0]->getTensorDesc().getDims();    
    
        float* src_data = inputs[0]->buffer();
        float* dst_data = outputs[0]->buffer();

        int _dsize = 1;
        for (size_t i = 0; i < in_dims.size(); i++)
            _dsize *= in_dims[i];

        for (int i = 0; i < _dsize; i++)
        {
            double __exp = exp(src_data[i]);
            dst_data[i] = (__exp + (1 / __exp)) / 2;
        }
        return OK;
    }

private:
};

REG_FACTORY_FOR(ImplFactory<coshImpl>, cosh);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine




