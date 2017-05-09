%% MergeInformation
%% Format
% [id]
% [gender] 0 1 2
% [Hat] 0:NA; 1:black; 2: blue; 3: brown; 4: green; 5: orenge; 6: red; 7:
% gray; 8: white; 9: yellow; 10: transparent; 11:multi-color;  99:can'tjudge;      
% [Glasses] 0:NA; 1:black; 2: blue; 3: brown; 4: green; 5: orenge; 6: red; 7:
% gray; 8: white; 9: yellow; 10: transparent; 11:multi-color;  99:can'tjudge;
% [Mask] 0:NA; 1:black; 2: blue; 3: brown; 4: green; 5: orenge; 6: red; 7:
% gray; 8: white; 9: yellow; 10: transparent; 11:multi-color;  99:can'tjudge;
% [num] 0,1,2,3
% [path] "collect1/[id].png","collect2/[id].png","collect3/[id].png"
% [bbox] [x,y,w,h],[x,y,w,h],[x,y,w,h]
%% load gender and color
load 'data.mat'
%% load bbox
A1 = fscanf(fopen('test1.txt'),'%f',[4 Inf]); A1=round(100*A1')/100;
A2 = fscanf(fopen('test2.txt'),'%f',[4 Inf]); A2=round(100*A2')/100;
A3 = fscanf(fopen('test3.txt'),'%f',[4 Inf]); A3=round(100*A3')/100;
%% start
fid = fopen('result.txt','wt');

for k = 1 : length(No)
    id = No(k);
    %% gender
    gender = Gendercode(k);
    %% color
    hat = Hat{k};  %, there should be number
%    string = strtrim(hat);  %count the number of colors
%     counter = 0;
%     while ~isempty(string)
%         [~, string] = strtok(string);
%         counter = counter + 1;
%     end
%     for i = 1 : counter
    string = strtrim(hat); %get the first color
    [string, left] = strtok(string);
        switch string   %judge the color
            case {'N.A.','N.A','NA','N. A.'}
                hat = 0;   
            case 'BLACK'
                hat = 1;
            case 'BLUE'
                hat = 2;
            case 'BROWN'
                hat = 3;
            case 'GREEN'
                hat = 4;
            case 'ORANGE'
                hat = 5;
            case {'RED','ROSE'}
                hat = 6;
            case {'GREY','GRAY','SILVER'}
                hat = 7;
            case 'WHITE'
                hat = 8;
            case {'YELLOW','TAN','GOLD','GOLDEN'}
                hat = 9;
            case 'TRANSPARENT'  
                hat = 10;
            case 'MULTI'
                hat = 11;         
            otherwise 
                hat = 99; %99 means can't judge
        end 
        if  ~isempty(left)
            hat = 11;
        end
    %end
    % Hat = 0;
    glass = Glass{k};%;, there should be number
    % Glass = 0;
    string = strtrim(glass); %get the first color
    [string, left] = strtok(string);
         switch string
            case {'N.A.','N.A','NA'}
                glass = 0;   
            case 'BLACK'
                glass = 1;
            case 'BLUE'
                glass = 2;
            case 'BROWN'
                glass = 3;
            case 'GREEN'
                glass = 4;
            case 'ORANGE'
                glass = 5;
            case {'RED','ROSE'}
                glass = 6;
            case {'GREY','GRAY','SILVER'}
                glass = 7;
            case 'WHITE'
                glass = 8;
            case {'YELLOW','TAN','GOLD','GOLDEN'}
                glass = 9;
            case 'TRANSPARENT'  
                glass = 10;
            case 'MULTI'
                glass = 11;
            otherwise 
                glass = 99;        
         end   
         if  ~isempty(left)
            glass = 11;
         end
    mask = Mask{k};%, there should be number
    % Mask = 0;
    string = strtrim(mask); %get the first color
    [string, left] = strtok(string);
         switch string
            case {'N.A.','N.A','NA'}
                mask = 0;   
            case 'BLACK'
                mask = 1;
            case 'BLUE'
                mask = 2;
            case 'BROWN'
                mask = 3;
            case 'GREEN'
                mask = 4;
            case 'ORANGE'
                mask = 5;
            case {'RED','ROSE'}
                mask = 6;
            case {'GREY','GRAY','SILVER'}
                mask = 7;
            case 'WHITE'
                mask = 8;
            case {'YELLOW','TAN','GOLD','GOLDEN'}
                mask = 9;
            case 'TRANSPARENT'
                mask = 10;           
            case 'MULTI'
                mask = 11;  
            otherwise 
                mask = 99;       
         end        
         if  ~isempty(left)
            mask = 11;
         end
    %% path and boxes
    path = {};boxes={};
    Name = [num2str(No(k)), '.png'];
    if exist(['modified_collect1/',Name], 'file') == 2
        path{end+1} = ['modified_collect1/',Name];        
        boxes{end+1}=A1(id,:);
    end
    if exist(['modified_collect2/',Name], 'file') == 2
        path{end+1}= ['modified_collect2/',Name];
        boxes{end+1}=A2(id,:);
    end
    if exist(['modified_collect3/',Name], 'file') == 2
        path{end+1} = ['modified_collect3/',Name];
        boxes{end+1}=A3(id,:);
    end
    num = length(path);
    if ~isempty(path)
        out = num2str([id, gender, hat, glass, mask, num]);        
        for l=1:num
            out = [out,' "', path{l},'" ',num2str(boxes{l}),' ' ];
        end
        disp(out)
        fprintf(fid,'%s \n',out);
    end
end

fclose(fid);
