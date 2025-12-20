# 1. Install audio dependencies
sudo apt-get update
sudo apt-get install portaudio19-dev alsa-utils pulseaudio

# 2. Add user to audio group
sudo usermod -a -G audio $USER
# Then logout and login again

# 3. Test microphone with ALSA
arecord -l  # List devices
arecord -d 3 test.wav  # Record 3 seconds

# 4. Check if PulseAudio is running
pulseaudio --check
pulseaudio --start  # If not running

# 5. Reinstall PyAudio
pip uninstall pyaudio
pip install pyaudio