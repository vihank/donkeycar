import tensorflow as tf
import matplotlib.pyplot as plt 

class AdvAttack:

    def __init__(self, kl, attack_freq):
        self.kl = kl
        self.attack_freq = attack_freq
        self.count=0

    
    def adversarial_pattern(self, image, label):
        image = tf.cast(image, tf.float32)
        pred = 0
        
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = self.kl.model.predict(image)
            pred = prediction
            loss = tf.keras.losses.MSE(label, prediction)
        
        gradient = tape.gradient(loss, image)
        
        signed_grad = tf.sign(gradient)
        
        plt.scatter(self.count, label, c='coral')
        plt.scatter(self.count, pred, c='lightblue')
        plt.show()
        return signed_grad

    def run(self, img, num_rec):
        if num_rec != None and num_rec % self.attack_freq == 0:
            img = img.reshape((1,) + img.shape)
            ang = self.kl.model.predict(img)
            
            grad = self.adversarial_pattern(img, ang).numpy()
            
            return img + (grad[0]*0.5 + 0.5), img