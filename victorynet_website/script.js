var latestModel = new Vue({
    el: '#latest-model',
    data: {
        modelDate: new Date(),
        modelName: "Test Vue Model Name",
        modelStatus: "null Lol",
        modelTrainingSize: 0,
        modelTestingSize: 0,
        modelEpoch: -1,
        modelLoss: 0.0000,
        modelValLoss: 0.00001,

        modelCfgEpochs: 0,
        modelCfgLearningRate: 0,
        modelCfgBatchSize: 0,
        modelCfgConv1Filter: 0,
        modelCfgConv1Kernal: 0,
        modelCfgConv2Filter: 0,
        modelCfgConv2Kernal: 0,
        modelCfgFullyConnected: 0,

        modelEnvAutoupdate: false,
        modelEnvLidarPoints: 0,
        modelEnvLidarNoise: 0,
        modelEnvLidarFailure: 0,
        modelEnvLidarMaxRange: 0
    }
})
var allModels = new Vue({
    el: '#all-models',
    data: {
        models: []
    },
    methods: {
        select: function (id) {
            var index = 0;
          
            for(var i=0;i<allModels.models.length;i++){
                
                if(allModels.models[i].info_name == id){
                    index = i;
                    console.log("Updating to: " + index)
                }
            }
           
            updateSelection(index);
            
        }
    }
})

var currentSelection = 0;

updateSelection = (id) =>{
    currentSelection = id;
    updateModels();
}

updateModels();

setInterval(() => {
    updateModels();
}, 30000)

function updateModels() {
    $.getJSON('https://us-central1-victory-net.cloudfunctions.net/fetch-models', function (data) {
        allModels.models = data;

        var latest = data[currentSelection];
        latestModel.modelName = latest['info_name']
        latestModel.modelDate = latest['info_date']
        latestModel.modelStatus = latest['info_status']

        latestModel.modelTrainingSize = latest['info_training_length']
        latestModel.modelTestingSize = latest['info_testing_length']

        latestModel.modelEpoch = latest['info_epoch'] + 1
        latestModel.modelLoss = latest['info_loss']
        latestModel.modelValLoss = latest['info_val_loss']

        latestModel.modelCfgEpochs = latest['cfg_epoch']
        latestModel.modelCfgLearningRate = latest['cfg_learning_rate']
        latestModel.modelCfgBatchSize = latest['cfg_batch_size']
        latestModel.modelCfgConv1Filter = latest['cfg_conv1_filter']
        latestModel.modelCfgConv1Kernal = latest['cfg_conv1_kernal']
        latestModel.modelCfgConv2Filter = latest['cfg_conv2_filter']
        latestModel.modelCfgConv2Kernal = latest['cfg_conv2_kernal']
        latestModel.modelCfgFullyConnected = latest['cfg_fully_connected']

        latestModel.modelEnvAutoupdate = latest['env_instantMode']
        latestModel.modelEnvLidarPoints = latest['env_lineNum']
        latestModel.modelEnvLidarNoise = latest['env_noise']
        latestModel.modelEnvLidarFailure = latest['env_dropout']
        latestModel.modelEnvLidarMaxRange = latest['env_maxRange']

        console.log(data)
    });

}