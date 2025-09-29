MODULE Comm
	CONST robtarget p10:=[[0,0,0],[0.643873,0.0206772,0.00067172,-0.764853],[0,0,-3,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget p20:=[[170.20,206.79,-208.46],[0.638087,-0.00205458,-0.00620429,-0.769936],[0,0,-3,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Home:=[[170.20,206.79,-208.46],[0.638087,-0.00205458,-0.00620429,-0.769936],[0,0,-3,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pCenter_Square:=[[48.45,42.53,-7.40],[0.778467,-0.00306691,-0.0001832,0.627678],[0,-1,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    
    
    VAR robtarget recvCoordi:=[[374.39,143.84,-94.15],[0.536346,-0.00756933,-0.0055927,-0.843946],[0,-1,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS wobjdata wobj1 := [FALSE, TRUE, "", [[307.824,-178.101,4.12927],[0.00124084,0.78718,0.616722,0.000969443]], [[0,0,0],[1,0,0,0]]];
	!!TASK PERS tooldata toolVacuum_old:=[TRUE,[[-91.8764,3.18803,186.338],[1,0,0,0]],[1,[20,20,25],[1,0,0,0],0,0,0]];
    TASK PERS tooldata toolVacuum:=[TRUE,[[-89.8475,2.72208,187.55],[1,0,0,0]],[1,[20,20,25],[1,0,0,0],0,0,0]];
	CONST robtarget pAway:=[[451.32,-82.15,113.60],[0.00472123,-0.55667,0.830604,-0.0138772],[-1,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget PosTemp:=[[89.97,100.74,-63.96],[0.0692979,0.00778393,0.00578711,-0.997549],[-1,-1,-2,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pApproachGrip:=[[299.94,312.47,-52.68],[0.0692137,0.00780942,0.00575971,-0.997555],[-1,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget pApproachPlace:=[[89.97,100.74,-63.96],[0.0692979,0.00778393,0.00578711,-0.997549],[-1,-1,-2,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	
    
    PROC main()
        !Robotserver; 
        RobotClient;
    ENDPROC  
    
    
    PROC Robotserver()
    VAR string receivedStringCoords;   !//Received string
    VAR string sendString;       !//Reply string
    VAR pos pickpos;
    VAR bool ok;

        !Create Server in controller
        ServerCreateAndConnect;
        
        !Handshake
        sendString:="request coords";
        TPWrite "Sending to camerasystem:"+sendString;
        SendMessage(sendString);
        receivedStringCoords:=ReciveMessage();
        TPWrite "Camera Answer:"+receivedStringCoords;	
        ok:=StrToval(receivedStringCoords,pickpos);
        WaitTime 1;
        sendString:="thanks";
        SendMessage(sendString);
        ServerCloseAndDisconnect;
        WaitTime 5;
    ENDPROC
    
    
PROC RobotClient()
    VAR string receivedStringCoords;   !//Received string
    VAR string sendString;       !//Reply string
    VAR pos pickpos;
    VAR bool ok;
    VAR num sendNum;
    
    VAR num x_coord;
    VAR num y_coord;
    VAR num orientation;
    VAR num shape;
    VAR num color;
    VAR string receivedMessage;
    
    VAR string temp;
    VAR num values{4};
    VAR string parts{4};
    VAR string remainder;
    VAR num i := 1;
    VAR num commaPos;
  
    !Get String from python
    ClientCreateAndConnect;
    receivedMessage:=ReciveMessage();
    waitTime(.25);
    
    !Ask for color with string
    TPReadNum color, receivedMessage;
    !color := 12345;
    
    !Send userinput to python
    TCPSendMessage(NumToStr(color,0));
    TPErase;
    
    !Get String from python
    receivedMessage:=ReciveMessage();
    !Ask for shape with string
    TPReadNum shape, TCPreceivedMessage;
    !shape := 12345;

    !Send userinput to python
    TPErase;
    SendMessage(NumToStr(shape,0)); 
        
    !Move robot out of camera view
    receivedMessage := ReciveMessage();
    WaitTime .5;
    MoveJ pAway, v1000,z100,toolVacuum\WObj:=wobj0;
    SendMessage("Moving out of frame finished");
    
    WHILE TRUE DO
    !Get String from python with positiondata x,y,orientation,shape
        receivedMessage := ReciveMessage();
        
        !If no shapes were found by phyton
        IF StrFind(receivedMessage,1, "No shapes found") = 1 THEN
            SendMessage("Thanks, then I will finish the process");
            EXIT;
        ENDIF
            
        !Extract data in variables    
        SplitString receivedMessage, ",", parts;
        FOR i FROM 1 TO 4 DO
            ok := StrToVal(parts{i}, values{i});
        ENDFOR
        
        ! Remap coords
        x_coord := values{1};
        y_coord := values{2};
        orientation := values{3};
        shape := values{4};
                
        ! Start program, pick place
        pickPlacePart x_coord,y_coord,orientation,shape;
        SendMessage("Pick and Place done");
      
        ! Quitting socket comm.
        receivedMessage := ReciveMessage();
        WaitTime .25;
        SendMessage("Done");
    ENDWHILE 
    
ENDPROC     


PROC SplitString(string source, string delimiter, INOUT string resultArray{*})
    VAR string temp;
    VAR num numOfValues := 4;
    VAR string fullString;
    VAR num i := 1;
    VAR num commaPos;
    VAR num startPos := 1;
    VAR bool ok;
    
    fullString := source;
    
    WHILE i <= numOfValues DO
        commaPos := StrFind(fullString, startPos, ",");
    
        IF commaPos > 0 THEN
            temp := StrPart(fullString, startPos, commaPos - startPos);
            resultArray{i} := temp;
            startPos := commaPos + 1;  ! N chste Runde beginnt nach dem Komma
        ENDIF
        
        i := i + 1;
    ENDWHILE
ENDPROC


PROC pickPlacePart(num x_coord,num y_coord,num orientation,num shape)
    VAR robtarget target;
    CONST num xOffset := 2;
    CONST num yOffset := 20;
    recvCoordi := pApproachGrip;
    recvCoordi.trans := [x_coord, y_coord, 0];
    
    ! Pick
    MoveJ pApproachGrip, v1000, z100, toolvacuum\WObj:=wobj1 ;
    MoveJ Offs(recvCoordi,xOffset,yOffset,-50),v1000,z50,toolVacuum\WObj:=wobj1; 
    MoveJ Offs(recvCoordi,xOffset,yOffset,-10),v1000,fine,toolVacuum\WObj:=wobj1;   
    SetDO dovalve1, 1;  !Activate vaccum
    MoveJ Offs(recvCoordi,xOffset,yOffset,-50),v1000,z50,toolVacuum\WObj:=wobj1;
    
    ! Get target location
    target := getPlacePos (shape, orientation);  !Need to know where to place the object
    
    ! Place
    MoveJ Offs(target,0,0,-50),v1000,z20,toolVacuum\WObj:=wobj1;   
    MoveL Offs(target,0,0,-10),v1000,fine,toolVacuum\WObj:=wobj1;  
    SetDO dovalve1, 0;  !Deactivate vaccum
    MoveJ Offs(target,0,0,-50),v1000,z20,toolVacuum\WObj:=wobj1;   

ENDPROC
    
FUNC robtarget getPlacePos(num shape, num orientation)
    
    VAR robtarget target := pApproachPlace;
    VAR num offset_angle := 0;
    VAR orient angles; 

    ! Definiere Offsets je nach Shape
    IF shape = 0 THEN
        target.trans := [157, 52.83, 0];  ! z.B. Triangle
        offset_angle:= 0;
    ELSEIF shape = 1 THEN
        target.trans := [41, 41, 0];      ! z.B. Square
        offset_angle:= 0; !-20, -45
    ELSEIF shape = 2 THEN
        target.trans := [144, 158, 0];    ! z.B. Hexagon
        offset_angle := 0;
    ELSEIF shape = 3 THEN
        target.trans := [36, 167, 0];     ! z.B. Circle
    ELSEIF shape = 4 THEN
        target.trans := [95, 100, 0];     ! z.B. Star
        offset_angle := 36; ! -18
    ELSE
        target.trans := [0, 0, -100];     ! Fallback
    ENDIF   
    
    target := RotZ(orientation + offset_angle, target);
    
    RETURN target;
ENDFUNC


FUNC robtarget RotZ(num angle_deg, robtarget target)
     VAR orient rotZOrient;
     VAR num anglex;
     VAR num angley;
     VAR num anglez;
     
     ! Get Euler angles from target
     anglex := EulerZYX(\X, target.rot);
     angley := EulerZYX(\Y, target.rot);
     anglez := EulerZYX(\Z, target.rot);
     
     ! Add orientation from object
     anglez := anglez + angle_deg;

     ! Go back to quaternions
     target.rot := OrientZYX(anglez, angley, anglex);
     RETURN target;
ENDFUNC


ENDMODULE
