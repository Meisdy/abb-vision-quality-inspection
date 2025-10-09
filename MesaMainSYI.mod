MODULE MesaMainSYI
    PERS wobjdata wobjTable:=[FALSE,TRUE,"",[[1324.77,-472.104,426.917],[0.999992,0.000929973,-0.000683746,0.00391044]],[[0,0,0],[1,0,0,0]]];
    PERS tooldata tGripper:=[TRUE,[[0.51636,-0.710444,275.457],[1,0,0,0]],[26,[80,0,180],[1,0,0,0],0,0,0]];
    PERS tooldata tPen:=[TRUE,[[-180.096,-2.75697,367.367],[1,0,0,0]],[22.7,[80,0,180],[1,0,0,0],0,0,0]];
    PERS robtarget pHomeTable:=[[87.36,384.09,290.00],[0.0022726,-0.0553747,-0.998463,0.000323689],[-1,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pHome:=[[1246.51,-18.26,1541.99],[0.00169958,-0.0540723,-0.998534,-0.00180012],[-1,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pApproachCamera:=[[371.01,634.71,95.07],[0.00046576,0.0554334,0.998462,-4.78765E-05],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pCamera:=[[440.00,640.00,24.20],[0.00360056,-0.055442,-0.998455,0.000647754],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pAway:=[[250.45,640.03,70.00],[0.00360485,-0.0554333,-0.998456,0.00064939],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];


    PROC main()
        !SetupPlcCom;           !Setup PLC com. (if necessary)
        !SetupVisionSystem;
        GetPart;               !Pickup Part and place to Camera System
        !VisionEvaluation;      !Evaluate Part using Vision Computer
        !ProcessPart            !Move Part away
        !Give info to PLC
        
        ResetSystem;


    ENDPROC
    
    
PROC SetupPlcCom()
    
    ! Test IOs
    SetDO QiABBStatus, 0;
    SetDO QiABBVerificationStatus, 0;
    
    
    
!    Declare digital input signals for PLC outputs mapped via Profinet
!    VAR signal IxStart;
!    VAR signal IxStop;    
!    IxStart := DI.Signal("IxStart");
!    IxStop := DI.Signal("IxStop");
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

    TPWrite "Confirm to continue with GetPart";
    Stop;
    ! Now the program waits until the operator presses "Continue" on the teach pendant
    
    ! Wait for starting signal
    !WaitDI IxABBStart, 1;
    
    ! Open Gripper if not already open
    !SET doValve1;
    !WaitTime 1;
    
    !Test Table positions
    

    
    ! Check where to get Part from
!    IF IxABBPickSelector = 1 THEN
!        ! Move J Approach AGV, maybe make  multiple ones depending on trajectory
!        ! Move L Exact Pickup Position
!        ! Close Gripper
!    ELSE
!        !
!    ENDIF
    
    ! Move to pickup position
    !MOVE J APPROACH 
        
    
    
    ! Get Start message and pickup location from PLC
    ! Move to right pickup location
    ! Pickup Part
    
    ! Place part below Camera and place it
    MoveJ pHomeTable, v2500, z100, tGripper\WObj:=wobjTable;            ! Go to table middle
    MoveJ Offs(pCamera, 0, 0, 50), v200, z15, tGripper\WObj:=wobjTable;  ! Go to approach camera pos
    MoveL pCamera, v50, fine, tGripper\WObj:=wobjTable;                 ! Go to camera dropoff pos
    OpenGripper;
    MoveL Offs(pCamera, 0, 0, 50), v200, z15, tGripper\WObj:=wobjTable;  ! Go to approach camera pos
    
    ! Move robot out of camera frame
    MoveL pAway, v2500, z50, tGripper\WObj:=wobjTable;              ! Go to trajectory approach pos for Camera
    TPWrite "Confirm to continue with GetPart";
    Stop;
    
    ! Pickup Part again and go to Home Table
    MoveL Offs(pCamera, 0, 0, 50), v200, z15, tGripper\WObj:=wobjTable;  ! Go to approach camera pos
    MoveL pCamera, v50, fine, tGripper\WObj:=wobjTable;                 ! Go to camera dropoff pos
    CloseGripper;
    MoveL Offs(pCamera, 0, 0, 50), v200, z15, tGripper\WObj:=wobjTable;  ! Go to approach camera pos
    MoveJ pHomeTable, v2500, z100, tGripper\WObj:=wobjTable;            ! Go to table middle

    

    ! Place part

ENDPROC

PROC VisionEvaluation()
    ! Send Request to Vision PC. Maybe change functions so Client is Initiator here... Or use quickfix in which Setup VIsion gets a new message and will just respond here.
ENDPROC


PROC RobotClient()
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

ENDPROC



PROC SplitString(string source, string delimiter, INOUT string resultArray{*})
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
ENDPROC



PROC pickPlacePart(num x_coord,num y_coord,num orientation,num shape)
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

ENDPROC



FUNC robtarget getPlacePos(num shape, num orientation)

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
ENDFUNC



FUNC robtarget RotZ(num angle_deg, robtarget target)
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
ENDFUNC

PROC ResetSystem()
    MoveJ pHome, v2500, z50, tool0\WObj:=wobj0;
    WaitTime\InPos ,.1;
    !Reset doValve1;


ENDPROC

PROC OpenGripper()
    SET doValve1;
    WaitTime .3;
ENDPROC

PROC CloseGripper()
    RESET doValve1;
    WaitTime .3;
ENDPROC



ENDMODULE
