import pygame
from pygame.locals import *
import sys
from random import randint

class TSVID:
    BLACK = 0,0,0
    WHITE = 255,255,255
    RED = 255,0,0
    GREEN = 0,255,0
    BLUE = 0,0,255
    GREY = 128,128,128
    DEEP_GREY = 40,40,40

    AXIS_WIDTH = 950
    AXIS_HEIGHT = 600
    TOP_NUM = 20 
    BAR_HEIGHT = AXIS_HEIGHT // (TOP_NUM*1.5)  * 9 // 10
    FADE_SPEED = 20


    def __init__(self, *, 
                 numstr = lambda num: str(num), 
                 lidtext = 'Aktualny lider: ', 
                 timetext = 'Lata jako lider: ',
                 interval = 30
                 ):
        self.lastk = None
        self.fadev = None
        self.store = None

        self.numstr = numstr
        self.TEXT_TOP1_TIME = timetext
        self.LIDER_ANNOTATION = lidtext
        self.DATE_INTERVAL = interval #ticks or frames 30=0.5s


    def HSV2RGB(H, S, V):
        if S == 0:
            return V,V,V
        else:
            H /= 60
            i = int(H)
            f = H - i
            a = int(V * (1-S))
            b = int(V * (1-S*f))
            c = int(V * (1-S*(1-f)))
            if i == 0:
                return V,c,a
            elif i == 1:
                return b,V,a
            elif i == 2:
                return a,V,c
            elif i == 3:
                return a,b,V
            elif i == 4:
                return c,a,V
            elif i == 5:
                return V,a,b

    def findname(ilist, value):
        for i in range(len(ilist)):
            if ilist[i].iname == value:
                return i
        return -1

    class Bar:
        def __init__(self, iname, itype, ivalue, DATE_INTERVAL):
            self.iname = iname
            self.itype = itype
            self.ivalue = ivalue
            self.lastvalue = ivalue
            self.color = TSVID.HSV2RGB(randint(0,359),0.7,230)
            self.lastwidth = 0
            self.rank = 21
            self.lastrank = 21
            self.DATE_INTERVAL = DATE_INTERVAL

        def get_pos(self, step, max_value):
            if step == -1:
                top = int(self.rank*TSVID.BAR_HEIGHT*1.5)
                if self.rank <= 20:
                    alpha = 255
                    show = True
                else:
                    alpha = 0
                    show = False
                value = self.ivalue
                width = (value/max_value)*TSVID.AXIS_WIDTH
                return top, width, value, alpha, show
            if self.rank != self.lastrank:
                start = self.lastrank*TSVID.BAR_HEIGHT*1.5
                end = self.rank*TSVID.BAR_HEIGHT*1.5
                top = int(start + (end-start)*(step/self.DATE_INTERVAL))
                if self.rank > 20 and self.lastrank <= 20:
                    alpha = 255 * (1-step/self.DATE_INTERVAL)
                    show = True
                elif self.rank <= 20 and self.lastrank > 20:
                    alpha = 255 * (step/self.DATE_INTERVAL)
                    show = True
                else:
                    if self.rank <= 20:
                        alpha = 255
                        show = True
                    else:
                        alpha = 0
                        show = False
            else:
                top = int(self.rank*TSVID.BAR_HEIGHT*1.5)
                if self.rank <= 20:
                    alpha = 255
                    show = True
                else:
                    alpha = 0
                    show = False
            start = self.lastvalue
            end = self.ivalue
            value = start + (end-start)*(step/self.DATE_INTERVAL)
            start = self.lastwidth
            end = (end/max_value)*TSVID.AXIS_WIDTH
            width = start + (end-start)*(step/self.DATE_INTERVAL)
            value = int(value)
            return top, width, value, alpha, show

    class BarList:
        def __init__(self, ilist, DATE_INTERVAL):
            self.data = ilist
            self.data.sort(key = lambda x: x.ivalue,reverse = True)
            for i in range(len(self.data)):
                self.data[i].lastrank = self.data[i].rank
                self.data[i].rank = i+1
            self.DATE_INTERVAL = DATE_INTERVAL

        def update(self, data, max_value):
            for each in data:
                temp = TSVID.findname(self.data,each['name'])
                if temp != -1:
                    self.data[temp].lastvalue = self.data[temp].ivalue
                    self.data[temp].lastwidth = (self.data[temp].ivalue/max_value)*TSVID.AXIS_WIDTH
                    self.data[temp].ivalue = each['value']
                else:
                    self.data.append(TSVID.Bar(each['name'],
                                        each['type'],
                                        each['value'],
                                        self.DATE_INTERVAL))
            self.data.sort(key = lambda x: x.ivalue,reverse = True)
            for i in range(len(self.data)):
                self.data[i].lastrank = self.data[i].rank
                self.data[i].rank = i+1
      
        
    def axis(self, max_value, min_limit, is_zero = True, min_value = 0):
        surface = pygame.surface.Surface((TSVID.AXIS_WIDTH+60,TSVID.AXIS_HEIGHT))
        surface.fill(TSVID.WHITE)
        font = pygame.font.SysFont('SimHei',15)

        if is_zero:
            if max_value >= min_limit:
                temp = len(str(int(max_value)))
                for i in (1,2,5,10):
                    if 6 <= max_value//(i*10**(temp-2)) <= 15:
                        kd = i*10**(temp-2)
                        num = max_value//kd
                        break
            else:
                temp = len(str(min_limit))
                for i in (1,2,5,10):
                    if 6 <= min_limit//(i*10**(temp-2)) <= 15:
                        kd = i*10**(temp-2)
                        num = min_limit//kd
                        break
                    
            if self.lastk != None and self.lastk < kd:
                self.fadev = TSVID.FADE_SPEED
            self.lastk = kd

            if self.fadev != 0:
                self.fadev -= 1
            
            if self.fadev != 0:
                if i == 5:
                    fkd = 2*10**(temp-2)
                else:
                    fkd = kd//2
                fnum = max_value//fkd

            if self.fadev != 0:
                for each in (x*fkd for x in range(fnum+1) if x*fkd not in (x*kd for x in range(num+1))):
                    temp = (each/max_value)*TSVID.AXIS_WIDTH+25
                    color = 255-127*(self.fadev/TSVID.FADE_SPEED)
                    color = color,color,color
                    pygame.draw.aaline(surface,color,(temp,0),(temp,TSVID.AXIS_HEIGHT-30))
                    tsur = font.render(self.numstr(each), True, color)
                    tsur_r = tsur.get_rect()
                    tsur_r.center = temp,TSVID.AXIS_HEIGHT-15
                    surface.blit(tsur,tsur_r)
            
            for each in (x*kd for x in range(num+1)):
                temp = (each/max_value)*TSVID.AXIS_WIDTH+25
                pygame.draw.aaline(surface,TSVID.GREY,(temp,0),(temp,TSVID.AXIS_HEIGHT-30))
                tsur = font.render(self.numstr(each), True, TSVID.GREY)
                tsur_r = tsur.get_rect()
                tsur_r.center = temp,TSVID.AXIS_HEIGHT-15
                surface.blit(tsur,tsur_r)

        return surface

    def top_bar(self, m_type, m_name, m_time):
        surface = pygame.surface.Surface((TSVID.AXIS_WIDTH+60,80))
        surface.fill(TSVID.WHITE)
        font = pygame.font.SysFont('SimHei',50)
        
        tsur = font.render(self.LIDER_ANNOTATION,True,TSVID.DEEP_GREY)
        surface.blit(tsur,(25,15))
        tsur = font.render(m_name,True,TSVID.DEEP_GREY)
        surface.blit(tsur,(25+int(TSVID.AXIS_WIDTH*0.3),15))
        tsur = font.render(self.TEXT_TOP1_TIME + str(m_time),True,TSVID.DEEP_GREY)
        tsur_r = tsur.get_rect()
        tsur_r.right = TSVID.AXIS_WIDTH+60
        tsur_r.top = 15
        surface.blit(tsur,tsur_r)
        return surface

    def bottom_date(dates):
        font = pygame.font.SysFont('SimHei',70)
        tsur = font.render(dates,True,TSVID.DEEP_GREY)
        return tsur

    def bar_graph(self, surface, pos, data, step):
        font = pygame.font.SysFont('SimHei',int(TSVID.BAR_HEIGHT)+2)
        font2 = pygame.font.SysFont('SimHei',int(TSVID.BAR_HEIGHT)+8)
        for each in self.store.data:
            top, width, value, alpha, show = each.get_pos(step,self.store.data[0].ivalue)
            if show:
                c = each.color[0],each.color[1],each.color[2],alpha
                pygame.draw.rect(surface,each.color,(pos[0]+1,pos[1]+top,width,TSVID.BAR_HEIGHT))
                tsur = font.render(each.iname,True,each.color)
                tsur_r = tsur.get_rect()
                tsur_r.right, tsur_r.top = pos[0]-5, pos[1]+top-1
                surface.blit(tsur,tsur_r)
                
                tsur = font.render(str(value),True,each.color)
                surface.blit(tsur,(pos[0] + width + 5, pos[1] + top-1))
                
        
    def make_bold(surface, tsur, rect):
        x, y = rect.left,rect.top
        surface.blit(tsur,(x-1,y-1))
        surface.blit(tsur,(x-1,y))
        surface.blit(tsur,(x-1,y+1))
        surface.blit(tsur,(x,y-1))
        surface.blit(tsur,(x,y))
        surface.blit(tsur,(x,y+1))
        surface.blit(tsur,(x+1,y-1))
        surface.blit(tsur,(x+1,y))
        surface.blit(tsur,(x+1,y+1))

    def create_video(self, data):
        ranks = {}
        for each in data:
            date = each[2]
            if date in ranks:
                ranks[date].append({'name':each[0],'type':each[0],'value':int(each[1])})
            else:
                ranks[date] = [{'name':each[0],'type':each[0],'value':int(each[1])}]
        data = ranks

        pygame.init()
        screen = pygame.display.set_mode((1280,720))
        clock = pygame.time.Clock()

        '''data = {'2010-08':[{'name':'A','type':'','value':1},
                        {'name':'B','type':'','value':2},
                        {'name':'C','type':'','value':3}],
                '2010-09':[{'name':'A','type':'','value':4},
                        {'name':'B','type':'','value':2},
                        {'name':'C','type':'','value':3}],
                '2010-10':[{'name':'A','type':'','value':5},
                        {'name':'B','type':'','value':3},
                        {'name':'C','type':'','value':4}],
                }'''

        self.store = TSVID.BarList([], self.DATE_INTERVAL)
        data_date = sorted(list(data))
        index = 0
        max_index = len(data_date)

        self.fadev = 0
        self.lastk = None
        frame = -1
        temp = sorted(data[data_date[0]],key = lambda x: x['value'],reverse=True)
        lastmaxv = temp[0]['value']
        self.store.update(data[data_date[0]],lastmaxv)
        top1 = 0
        lasttop1 = ''
        
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

            if index == -1:
                clock.tick(60)
                continue

            screen.fill(TSVID.WHITE)

            frame += 1
            if frame == self.DATE_INTERVAL+1 and index != -1:
                frame = 0
                index += 1

            if frame == 0:
                top1 += 1
                if index == max_index:
                    self.store.update(data[data_date[max_index-1]],self.store.data[0].ivalue)
                else:
                    self.store.update(data[data_date[index]],self.store.data[0].ivalue)


            maxv = self.store.data[0].ivalue
            maxv = int(lastmaxv + (maxv-lastmaxv)*(frame/self.DATE_INTERVAL))
            if frame == self.DATE_INTERVAL:
                lastmaxv = self.store.data[0].ivalue
            temp = self.axis(maxv, 10)
            axistemp = temp.get_rect()
            axistemp.left, axistemp.top = 150, 80
            axistemp = axistemp.right,axistemp.bottom
            screen.blit(temp,(150,80))

            if lasttop1 != self.store.data[0].iname:
                lasttop1 = self.store.data[0].iname
                top1 = 0
            ttype, tname = self.store.data[0].itype, lasttop1
            temp = self.top_bar(ttype,tname,top1)
            screen.blit(temp,(150,0))

            if index == max_index:
                temp = TSVID.bottom_date(data_date[max_index-1])
            else:
                temp = TSVID.bottom_date(data_date[index])
            temp_r = temp.get_rect()
            temp_r.center = axistemp[0]-35, 0
            temp_r.bottom = axistemp[1]-30
            screen.blit(temp,temp_r)

            if index == max_index:
                self.bar_graph(screen,(175,80),data[data_date[max_index-1]],-1)
                index = -1
            else:
                self.bar_graph(screen,(175,80),data[data_date[index]],frame)

            pygame.display.flip()
            clock.tick(60)
