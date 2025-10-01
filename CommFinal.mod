MODULE Comm
    CONST robtarget pHome:=[[1443.05,-78.10,1374.39],[0.041699,0.0440525,0.998159,-0.000109628],[-1,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];


    PROC main()
        !SetupPlcCom;           !Setup PLC com. (if necessary)
        SetupVisionSystem;
        !GetPart;               !Pickup Part and place to Camera System
        !VisionEvaluation;      !Evaluate Part using Vision Computer
        !ProcessPart            !Move Part away
        !Give info to PLC
        !Home system


    ENDPROC


PROC SetupPlcCom()
    ! Here is Setup for PLC if needed
ENDPROC

PROC SetupVisionSystem()
    VAR string receivedMessage;

    !Create and Connect TCP Client to Server
    ClientCreateAndConnect;

    !Wait for Server message
    receivedMessage:=TCPReceiveMessage();
    TPWrite("Server message: "+receivedMessage);
    waitTime(.25);

    !Respond to Server
    TCPSendMessage("robot ready");
ENDPROC

PROC GetPart()

    ! Get Start message and pickup location from PLC
    ! Move to right pickup location
    ! Pickup Part
    ! Move to Camera System
    ! (Place Part)

ENDPROC

PROC VisionEvaluation()
    ! Send Request to Vision PC. Maybe change functions so Client is Initiator here... Or use quickfix in which Setup VIsion gets a new message and will just respond here.
ENDPROC


!PROC RobotClient()
!    VAR string receivedStringCoords;   !//Received string
!    VAR string sendString;       !//Reply string
!    VAR pos pickpos;
!    VAR bool ok;
!    VAR num sendNum;

!    VAR num x_coord;
!    VAR num y_coord;
!    VAR num orientation;
!    VAR num shape;
!    VAR num color;
!    VAR string receivedMessage;

!    VAR string temp;
!    VAR num values{4};
!    VAR string parts{4};
!    VAR string remainder;
!    VAR num i := 1;
!    VAR num commaPos;

!    !Get String from python
!    ClientCreateAndConnect;
!    receivedMessage:=TCPReceiveMessage();
!    waitTime(.25);

!    !Ask for color with string
!    TPReadNum color, receivedMessage;
!    !color := 12345;

!    !Send userinput to python
!    TCPSendMessage(NumToStr(color,0));
!    TPErase;

!    !Get String from python
!    receivedMessage:=TCPReceiveMessage();
!    !Ask for shape with string
!    TPReadNum shape, receivedMessage;
!    !shape := 12345;

!    !Send userinput to python
!    TPErase;
!    TCPSendMessage(NumToStr(shape,0));

!    !Move robot out of camera view
!    receivedMessage := TCPReceiveMessage();
!    WaitTime .5;
!    MoveJ pAway, v1000,z100,toolVacuum\WObj:=wobj0;
!    TCPSendMessage("Moving out of frame finished");

!    WHILE TRUE DO
!    !Get String from python with positiondata x,y,orientation,shape
!        receivedMessage := TCPReceiveMessage();

!        !If no shapes were found by phyton
!        IF StrFind(receivedMessage,1, "No shapes found") = 1 THEN
!            TCPSendMessage("Thanks, then I will finish the process");
!            EXIT;
!        ENDIF

!        !Extract data in variables
!        SplitString receivedMessage, ",", parts;
!        FOR i FROM 1 TO 4 DO
!            ok := StrToVal(parts{i}, values{i});
!        ENDFOR

!        ! Remap coords
!        x_coord := values{1};
!        y_coord := values{2};
!        orientation := values{3};
!        shape := values{4};

!        ! Start program, pick place
!        pickPlacePart x_coord,y_coord,orientation,shape;
!        TCPSendMessage("Pick and Place done");

!        ! Quitting socket comm.
!        receivedMessage := TCPReceiveMessage();
!        WaitTime .25;
!        TCPSendMessage("Done");
!    ENDWHILE

!ENDPROC



!PROC SplitString(string source, string delimiter, INOUT string resultArray{*})
!    VAR string temp;
!    VAR num numOfValues := 4;
!    VAR string fullString;
!    VAR num i := 1;
!    VAR num commaPos;
!    VAR num startPos := 1;
!    VAR bool ok;

!    fullString := source;

!    WHILE i <= numOfValues DO
!        commaPos := StrFind(fullString, startPos, ",");

!        IF commaPos > 0 THEN
!            temp := StrPart(fullString, startPos, commaPos - startPos);
!            resultArray{i} := temp;
!            startPos := commaPos + 1;  ! N chste Runde beginnt nach dem Komma
!        ENDIF

!        i := i + 1;
!    ENDWHILE
!ENDPROC



!PROC pickPlacePart(num x_coord,num y_coord,num orientation,num shape)
!    VAR robtarget target;
!    CONST num xOffset := 2;
!    CONST num yOffset := 20;
!    recvCoordi := pApproachGrip;
!    recvCoordi.trans := [x_coord, y_coord, 0];

!    ! Pick
!    MoveJ pApproachGrip, v1000, z100, toolvacuum\WObj:=wobj1 ;
!    MoveJ Offs(recvCoordi,xOffset,yOffset,-50),v1000,z50,toolVacuum\WObj:=wobj1;
!    MoveJ Offs(recvCoordi,xOffset,yOffset,-10),v1000,fine,toolVacuum\WObj:=wobj1;
!    SetDO dovalve1, 1;  !Activate vaccum
!    MoveJ Offs(recvCoordi,xOffset,yOffset,-50),v1000,z50,toolVacuum\WObj:=wobj1;

!    ! Get target location
!    target := getPlacePos (shape, orientation);  !Need to know where to place the object

!    ! Place
!    MoveJ Offs(target,0,0,-50),v1000,z20,toolVacuum\WObj:=wobj1;
!    MoveL Offs(target,0,0,-10),v1000,fine,toolVacuum\WObj:=wobj1;
!    SetDO dovalve1, 0;  !Deactivate vaccum
!    MoveJ Offs(target,0,0,-50),v1000,z20,toolVacuum\WObj:=wobj1;

!ENDPROC



!FUNC robtarget getPlacePos(num shape, num orientation)

!    VAR robtarget target := pApproachPlace;
!    VAR num offset_angle := 0;
!    VAR orient angles;

!    ! Definiere Offsets je nach Shape
!    IF shape = 0 THEN
!        target.trans := [157, 52.83, 0];  ! z.B. Triangle
!        offset_angle:= 0;
!    ELSEIF shape = 1 THEN
!        target.trans := [41, 41, 0];      ! z.B. Square
!        offset_angle:= 0; !-20, -45
!    ELSEIF shape = 2 THEN
!        target.trans := [144, 158, 0];    ! z.B. Hexagon
!        offset_angle := 0;
!    ELSEIF shape = 3 THEN
!        target.trans := [36, 167, 0];     ! z.B. Circle
!    ELSEIF shape = 4 THEN
!        target.trans := [95, 100, 0];     ! z.B. Star
!        offset_angle := 36; ! -18
!    ELSE
!        target.trans := [0, 0, -100];     ! Fallback
!    ENDIF

!    target := RotZ(orientation + offset_angle, target);

!    RETURN target;
!ENDFUNC



!FUNC robtarget RotZ(num angle_deg, robtarget target)
!     VAR orient rotZOrient;
!     VAR num anglex;
!     VAR num angley;
!     VAR num anglez;

!     ! Get Euler angles from target
!     anglex := EulerZYX(\X, target.rot);
!     angley := EulerZYX(\Y, target.rot);
!     anglez := EulerZYX(\Z, target.rot);

!     ! Add orientation from object
!     anglez := anglez + angle_deg;

!     ! Go back to quaternions
!     target.rot := OrientZYX(anglez, angley, anglex);
!     RETURN target;
!ENDFUNC



ENDMODULE
