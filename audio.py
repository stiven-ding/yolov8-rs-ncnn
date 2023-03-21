import threading
import time

import pyglet
pyglet.options["headless"] = True
import pyglet.media

from multiprocessing.connection import Listener

class AudioQueue(threading.Thread):

    def __init__(self, obstacle):
        threading.Thread.__init__(self)
        self.dist = 10

        import pyglet
        pyglet.options["headless"] = True
        import pyglet.media

    
    def update(self, dist, x, y, name = ''):
        self.dist = dist
        self.x = x
        self.y = y

        self.name = name

    def run(self):
        while True:
            if self.dist < 5:
                self.generate_sound()
            else:
                time.sleep(0.5)

    def generate_sound(self):
        #global player 
        player = pyglet.media.Player()
        player.play()
        
        audio_x = (self.x-320)/213 # -1.5 to 1.5
        audio_y = (self.y-240)/240
        audio_z = self.dist/3

        audio_dur = min(self.dist/3, 0.10)
        idle_dur = self.dist

        freq = 400 + audio_z*100

        wave = pyglet.media.synthesis.Triangle(audio_dur, freq)
        player.position = (audio_x, audio_y, audio_z)
        
        player.queue(wave)
        
        time.sleep(audio_dur+idle_dur)
        #player.delete()

def main():
    audio_queue = AudioQueue(None)
    audio_queue.start()

    address = ('localhost', 55777)     
    listener = Listener(address, authkey=b'secret password')
            
    while True:
        print('waiting for connection')
        conn = listener.accept()
        print('connection accepted from', listener.last_accepted)
        while True:
            try:
                msg = conn.recv()
                print(msg)
                
                obs = msg.split(',')
                dist, x, y, name = float(obs[0]), float(obs[1]), float(obs[2]), obs[3]

                audio_queue.update(dist, x, y, name)

                if msg == 'close':
                    conn.close()
                    break
            except:
                print('disconnected')
                break

if __name__ == '__main__':
    main()