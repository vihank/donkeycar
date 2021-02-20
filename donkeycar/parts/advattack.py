import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class AdvAttack:

    def __init__(self, kl, attack_freq=5, eps=0.2):
        self.kl = kl
        self.attack_freq = attack_freq
        self.attackCount = 0
        self.epsilon = eps
        print('with attacks!')

    
    def adversarial_pattern(self, image, label):
        image = tf.cast(image, tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = self.kl.model(image)
            loss = tf.keras.losses.MSE(label, prediction)
        
        gradient = tape.gradient(loss, image)
        
        signed_grad = tf.sign(gradient)
        
        return signed_grad

    def run(self, img, num_rec):
        if num_rec != None and num_rec % self.attack_freq == 0:
            self.attackCount+=1
            image = img.reshape((1,) + img.shape)
            ang = self.kl.model.predict(image)
            
            perturbation = self.adversarial_pattern(image, ang).numpy()
            perturb = ((perturbation[0]*0.5 + 0.5)*255)-50
            adv_img = np.clip(img + (perturb*self.epsilon), 0, 255)
            adv_img = adv_img.astype(int)

            '''
            # Sanity check
            adv_ang = self.kl.run(adv_img)
            plt.scatter(self.attackCount, ang[0][0], c='red')
            plt.scatter(self.attackCount, adv_ang[0], c='blue')
            if (self.attackCount % 5) == 0:
                plt.show()
                
            _ , axs = plt.subplots(3, 1)
            axs[0].imshow(perturb)
            axs[1].imshow(img)
            axs[2].imshow(adv_img)
            plt.show()'''
            return adv_img, img, ang[0][0]
        else: 
            return img, None, None

    def __call__(self, img):
        self.attackCount+=1
        image = img.reshape((1,) + img.shape)
        ang = self.kl.model.predict(image)

        perturbation = self.adversarial_pattern(image, ang).numpy()
        perturb = ((perturbation[0]*0.5 + 0.5)*255)-50
        adv_img = np.clip(img + (perturb*self.epsilon), 0, 255)
        adv_img = adv_img.astype(int)
        adv_ang = self.kl.model.predict(adv_img)
        return ang, adv_ang