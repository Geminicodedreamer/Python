import wave

# just suit for 16bit if 32bit set 0-3


def bytesarray2str(temp):
    # print(temp,temp[0],temp[1])
    temp_str = '0x{:02X}{:02X}'.format(temp[1], temp[0])
    # print(temp_str)
    return temp_str


def get_wav_file(path):

    print(path)

    wave_read = wave.open(path, "r")
    nchannels, bits, samplerate, nframes, comptype, compname = wave_read.getparams()

    print(nchannels, bits, samplerate, nframes, comptype, compname)

    print("channel    = ", nchannels)
    print("bits       = ", bits*8)   # bits is one byte
    print("samplerate = ", samplerate)
    print("nframes    = ", nframes)         # wave len

    # h file name
    h_name = 'c'+str(nchannels)+'_b'+str(bits*8)+'_s'+str(samplerate)
    h_path = h_name+'.h'
    print(h_path)

    max_list = nframes

    with open(h_path, 'w+') as outFile:
        outFile.write("#include <stdint.h>\n")
        outFile.write("int16_t "+h_name+"[] = {\n")

        for i in range(max_list):
            temp = wave_read.readframes(1)
            xx_str = bytesarray2str(temp)
            # print(xx_str)
            outFile.write(xx_str)
            outFile.write(',')
            # print('i=',i%4)
            if i % 8 == 7:
                # print('xxxx')
                outFile.write('\n')
        outFile.write('0')
        outFile.write('};')

    outFile.close()

    wave_read.close()
    return


wav_path = "C:\\Users\\DELL\\Desktop\\学习资料\\arduino\\audio\\sound1.wav"
get_wav_file(wav_path)
