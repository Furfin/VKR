from keras.api.callbacks import Callback
from prometheus_client import Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
import uuid

registry = CollectorRegistry()


accuracy = Gauge('ak_trial_accuracy', 'Training accuracy', 
                    ['model_name', 'trial_uuid', 'epoch', 'model_type'], registry=REGISTRY)
loss = Gauge('ak_trial_loss', 'Training loss', 
                ['model_name', 'trial_uuid', 'epoch', 'model_type'], registry=REGISTRY)
val_accuracy = Gauge('ak_trial_val_accuracy', 'Validation accuracy', 
                        ['model_name', 'trial_uuid', 'epoch', 'model_type'], registry=REGISTRY)
val_loss = Gauge('ak_trial_val_loss', 'Validation loss', 
                    ['model_name', 'trial_uuid', 'epoch', 'model_type'], registry=REGISTRY)


class AutoKerasMetricsCallback(Callback):
    def __init__(self, model_name, model_type):
        super().__init__()
        self.model_name = model_name
        self.model_type = model_type
        self.trial_uuid = str(uuid.uuid4())
    
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))
        print(logs.items())
        logs = logs or {}
        loss.labels(
            model_name=self.model_name,
            trial_uuid=self.trial_uuid,
            epoch=str(epoch),
            model_type=self.model_type
        ).set(logs.get('loss', 0))
        
        val_loss.labels(
            model_name=self.model_name,
            trial_uuid=self.trial_uuid,
            epoch=str(epoch),
            model_type=self.model_type
        ).set(logs.get('val_loss', 0))
        
        if self.model_type == 'classification':
            accuracy.labels(
                model_name=self.model_name,
                trial_uuid=self.trial_uuid,
                epoch=str(epoch),
                model_type=self.model_type
            ).set(logs.get('accuracy', 0))
            
            val_accuracy.labels(
                model_name=self.model_name,
                trial_uuid=self.trial_uuid,
                epoch=str(epoch),
                model_type=self.model_type
            ).set(logs.get('val_accuracy', 0))