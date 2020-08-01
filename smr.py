#    ___ _  _ ___ __  __ _____   __
#   / __| || |_ _|  \/  | _ \ \ / /
#  | (__| __ || || |\/| |  _/\ V / 
#   \___|_||_|___|_|  |_|_|   |_| 
# 
# Copyright (c) 2020 Mihaly Kollo. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.

import datetime
from tabulate import tabulate
from scipy import histogram

class Smr():
    
    def __init__(self, file_name):
        self.file_name = file_name
        self.open_file()
        self.read_header()
        self.channels = []
        for i in range(1,33):
            self.channels.append(Channel(self, i))
        self.close_file()
               
    def open_file(self):
        self.fid = open(self.file_name, 'rb')
        
    def close_file(self):
        self.fid.close()
  
    def channel_list(self):        
        table = []
        for ch in self.channels:
            if ch.kind is not None:               
                table.append([ch.channel_no, ch.kind, ch.title, ch.comment, ch.units])
        print(tabulate(table,["No", "Kind", "Title", "Comment", "Unit"],tablefmt = "fancy_grid"))                
    
    def get_channel_data(self, chan):        
        i=[ch.channel_no for ch in self.channels].index(chan)
        with open(self.file_name, 'rb') as self.fid:
            if self.channels[i].kind == 'Adc':
                return self.channels[i].get_adc_data()
            elif self.channels[i].kind == 'Level':
                return self.channels[i].get_level_data()
            elif self.channels[i].kind in ['Event-','Event+','Marker','TextMark']:
                return self.channels[i].get_marker_data()            
            else:
                return None
        
    def read_header(self):
        self.system_id = self.read_short()
        self.copyright = self.read_string(10)
        self.creator = self.read_string(8)
        self.us_per_time = self.read_short()
        self.time_per_adc = self.read_short()
        self.file_state = self.read_short()
        self.first_data = self.read_long()
        self.channels = self.read_short()
        self.chan_size = self.read_short()
        self.extra_data = self.read_short()
        self.buffer_size = self.read_short()
        self.os_format = self.read_short()
        self.max_ftime = self.read_long()
        self.dtime_base = self.read_double()
        self.rec_datetime = self.read_datetime()
        pad  =  self.fid.read(52)
        self.comment = self.read_comment()
        
    def read_byte(self):
        return np.fromfile(self.fid, dtype='B', count=1)[0]
    
    def read_short(self):
        return np.fromfile(self.fid, dtype=np.uint16, count=1)[0]
    
    def read_shorts(self, n):
        return np.fromfile(self.fid, dtype=np.uint16, count=n)

    def read_long(self):
        return np.fromfile(self.fid, dtype=np.uint32, count=1)[0]
            
    def read_longs(self, n):
        return np.fromfile(self.fid, dtype=np.uint32, count=n)  
      
    def read_float(self):
        return np.fromfile(self.fid, dtype='<f', count=1)[0]
    
    def read_double(self):
        return np.fromfile(self.fid, dtype='<d', count=1)[0]
    
    def read_string(self, length):
        return np.fromfile(self.fid, dtype='|S1', count=length).tobytes().decode()
          
    def read_datetime(self):
        da = np.fromfile(self.fid, dtype=np.byte, count=6)
        year = self.read_short()        
        return datetime.datetime(year, da[5], da[4], da[3], da[2], da[1], da[0])

    def read_var_string(self, length):
        pointer  =  self.fid.tell()
        string_length = np.fromfile(self.fid, dtype=np.byte, count=1)[0]
        string = self.fid.read(string_length).decode()
        pointer  =  pointer + length
        self.fid.seek(pointer)
        return string
    
    def read_comment(self):
        comment = {}
        for i in range(1, 6):
            comment[i] = self.read_var_string(80)
        return comment        
    

class Channel():
    
    CHANNEL_CODES = [None,'Adc','Event-','Event+','Level','Marker','AdcMark','RealMark','TextMark','RealWave']
        
    def __init__(self, smr, channel_no):     
        self.smr = smr        
        base  =  512 + (140*(channel_no-1))
        smr.fid.seek(base)
        self.channel_no = channel_no
        self.del_size = smr.read_short()
        self.next_del_block = smr.read_long()
        self.first_block = smr.read_long()
        self.last_block = smr.read_long()
        self.blocks = smr.read_short()
        self.n_extra = smr.read_short()
        self.pre_trig = smr.read_short()
        self.free0 = smr.read_short()
        self.py_sz = smr.read_short()
        self.max_data = smr.read_short()
        self.comment = smr.read_var_string(72)
        self.max_chan_time = smr.read_long()
        self.chan_div = smr.read_long()
        self.phy_chan = smr.read_short()
        self.title = smr.read_var_string(14)
        self.kind = self.CHANNEL_CODES[smr.fid.read(1)[0]]
        smr.read_byte()
        if self.kind in ['Adc', 'AdcMark']:
            self.scale = smr.read_float()
            self.offset = smr.read_float()
            self.units = smr.read_var_string(6)
            self.interleave = smr.read_long()
        elif self.kind in ['RealMark', 'RealWave']:
            self.units = None
            self.min = smr.read_float()
            self.max = smr.read_float()
            self.units = smr.read_var_string(6)
            self.interleave = smr.read_long()
        elif self.kind in ['Level']:
            self.units = None
            self.init_low = smr.read_byte()
            self.next_low = smr.read_byte()
        else:
            self.units = None
        if self.kind=='Adc':
            self.get_block_header()
            
    def get_block_header(self):
        if self.kind is not None:
            self.block_headers = np.zeros([6, self.blocks], int)
            self.smr.fid.seek(self.first_block)
            self.smr.read_long()
            pointer = self.smr.read_long()
            self.block_pointers = []
            self.block_next_pointers = []
            self.block_start_times = []
            self.block_end_times = []
            self.block_channels = []
            self.block_items = []
            for i in range(1,self.blocks):
                self.smr.fid.seek(pointer)
                self.block_pointers.append(self.smr.read_long())
                pointer = self.smr.read_long()
                self.block_next_pointers.append(pointer)                
                self.block_start_times.append(self.smr.read_long())
                self.block_end_times.append(self.smr.read_long())        
                self.block_channels.append(self.smr.read_short())
                self.block_items.append(self.smr.read_short()) 

                
    def get_adc_data(self): # only works with continuous data!                        
        data  =  np.zeros(sum(self.block_items), np.short)
        n  =  0
        for i in range(len(self.block_pointers)):
            self.smr.fid.seek(self.block_pointers[i] + 20)
            data[n:n+self.block_items[i]] = self.smr.read_shorts(self.block_items[i])
            n += self.block_items[i]        
        self.start = self.block_start_times[0]
        self.stop = self.block_end_times[-1]        
        np.set_printoptions(precision = 3,suppress = True)                
        times=np.linspace(0,self.max_chan_time,num=data.shape[0])/100000    
        return np.vstack((times,data.astype('double') * self.scale / 6553.6 + self.offset))

    def get_level_data(self):
        self.smr.fid.seek(self.first_block)
        print(self.smr.fid.read(100).hex())
        

        pass
    
    def get_marker_data(self):
        pass