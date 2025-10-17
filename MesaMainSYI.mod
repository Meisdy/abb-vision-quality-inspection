MODULE MesaMainSYI
    PERS wobjdata wobjTable:=[FALSE,TRUE,"",[[1324.77,-472.104,426.917],[0.999992,0.000929973,-0.000683746,0.00391044]],[[0,0,0],[1,0,0,0]]];
    PERS tooldata tGripper:=[TRUE,[[0.51636,-0.710444,275.457],[1,0,0,0]],[26,[80,0,180],[1,0,0,0],0,0,0]];
    PERS tooldata tPen:=[TRUE,[[-180.096,-2.75697,367.367],[1,0,0,0]],[22.7,[80,0,180],[1,0,0,0],0,0,0]];
    PERS robtarget pHomeTable:=[[87.36,384.09,290.00],[0.0022726,-0.0553747,-0.998463,0.000323689],[-1,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pHome:=[[1246.17,-13.35,1540.00],[0.00170027,-0.0511174,-0.99869,-0.00183437],[-1,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pApproachCamera:=[[371.01,634.71,95.07],[0.00046576,0.0554334,0.998462,-4.78765E-05],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pCamera:=[[440.00,640.00,24.20],[0.00360056,-0.055442,-0.998455,0.000647754],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pAway:=[[250.45,640.03,70.00],[0.00360485,-0.0554333,-0.998456,0.00064939],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS wobjdata wobjConveyor:=[FALSE,TRUE,"",[[225.01,1139.56,326.568],[0.999943,-0.010696,-0.000444899,0.000237186]],[[0,0,0],[1,0,0,0]]];
    CONST robtarget pApproachConveyor:=[[329.56,101.68,179.90],[0.00508886,-0.676704,0.736203,0.00713396],[0,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pConveyor:=[[329.53,107.82,-20.60],[0.0050216,-0.671333,0.741104,0.00719529],[0,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pTrajectory:=[[787.71,775.77,904.87],[0.00148502,0.494751,-0.869034,0.000424912],[0,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    ! Add AGV Positions here later


    CONST bool USE_PLC := FALSE;
    CONST bool USE_VISION := True;
    PERS intnum visionResult := 2;

    
    PROC main()
        TPErase;
        
        IF NOT USE_PLC THEN
            TPWrite "Ignore PLC ON";
        ENDIF
        IF NOT USE_VISION THEN
            TPWrite "Ignore Vision ON";
        ENDIF
        
        !SetupPlcCom;           !Setup PLC com. (if necessary)
        SetupVisionSystem;
        GetPart;               !Pickup Part and place to Camera System
        VisionEvaluation;      !Evaluate Part using Vision Computer
        ProcessPart;            !Move Part away
        !Give info to PLC
        ResetSystem;

    ENDPROC
    
    
PROC SetupPlcCom()
    
    IF USE_PLC THEN
        SetDO QiABBStatus, 0;
        SetDO QiABBVerificationStatus, 0;
    ENDIF
    
    ! Test IOs

ENDPROC

PROC SetupVisionSystem()
    VAR string receivedMessage;

    IF USE_VISION THEN

        !Create and Connect TCP Client to Server
        ClientCreateAndConnect;
    
        !Wait for Server message
        receivedMessage:=TCPReceiveMessage();
        TPWrite("Server message: "+receivedMessage);
        waitTime(.25);
    
        !Respond to Server
        TCPSendMessage("robot ready");
        
    ENDIF
ENDPROC

PROC GetPart()
    VAR intnum partSource := 0;

    ! Define where to get the part from and wait for Execution signal
    IF NOT USE_PLC THEN
        TPReadFK partSource, "Choose part source?", "AGV", "Conveyor",stEmpty, stEmpty, stEmpty;
    ELSE
        partSource := IxABBPickSelector;
        WaitDI IxABBStart, 1;
    ENDIF
    
    ! Open Gripper if not already open
    OpenGripper;
    
    ! Check where to get Part from and move to pickup position
    IF partSource = 1 THEN ! Move to AGV and pickup part
        TPWrite "Not programed yet";
        EXIT;
!        MoveJ pApproachAGV, v2500, z100, tGripper\WObj:=wobjAGV;  ! Go to approach AGV pos
!        MoveL Offs(pAGV, 0, 0, 50), v200, z15, tGripper\WObj:=wobjAGV;  ! Go to approach AGV pos using offset
!        MoveL pAGV, v50, fine, tGripper\WObj:=wobjAGV;                 ! Go to AGV pickup pos
!        CloseGripper;
!        MoveL Offs(pAGV, 0, 0, 50), v200, z15, tGripper\WObj:=wobjAGV;  ! Go back to approach AGV pos using offset
!        MoveJ pApproachAGV, v2500, z100, tGripper\WObj:=wobjAGV;  ! Go to approach AGV pos
!        MoveJ pTrajectory, v2500, z100, tGripper\WObj:=wobj0;     ! Move to trajectory approach pos back to table

    ELSEIF partSource = 2 THEN ! Move to CONVEYOR and pickup part
        MoveJ pTrajectory, v2500, z100, tGripper;                       ! Move to trajectory approach pos back to table
        MoveJ pApproachConveyor, v2500, z100, tGripper\WObj:=wobjConveyor;  ! Go to approach Conveyor pos
        MoveL Offs(pConveyor, 0, 0, 50), v200, z15, tGripper\WObj:=wobjConveyor;  ! Go to approach Conveyor pos using offset
        MoveL pConveyor, v50, fine, tGripper\WObj:=wobjConveyor;                 ! Go to Conveyor pickup pos
        CloseGripper;
        MoveL Offs(pConveyor, 0, 0, 50), v200, z15, tGripper\WObj:=wobjConveyor;  ! Go back to approach Conveyor pos using offset
        MoveJ pApproachConveyor, v2500, z100, tGripper\WObj:=wobjConveyor;  ! Go to approach Conveyor pos
        MoveJ pTrajectory, v2500, z100, tGripper;                       ! Move to trajectory approach pos back to table
    ELSE 
        TPWrite "Error; Faulty part source value";
        EXIT;
    ENDIF
    
    ! Place part below Camera and place it
    MoveJ pHomeTable, v2500, z100, tGripper\WObj:=wobjTable;            ! Go to table middle
    MoveJ Offs(pCamera, 0, 0, 50), v200, z15, tGripper\WObj:=wobjTable;  ! Go to approach camera pos
    MoveL pCamera, v50, fine, tGripper\WObj:=wobjTable;                 ! Go to camera dropoff pos
    OpenGripper;
    MoveL Offs(pCamera, 0, 0, 50), v200, z15, tGripper\WObj:=wobjTable;  ! Go to approach camera pos
    
    ! Move robot out of camera frame
    MoveL pAway, v2500, z50, tGripper\WObj:=wobjTable;              ! Go to trajectory approach pos for Camera
        
ENDPROC

! Send request to Vision PC to evaluate the part
PROC VisionEvaluation()
    VAR string answer;
    
    IF NOT USE_VISION THEN
        TPReadFK visionResult, "Choose vision result?", "Complete (AGV)", "Incomplete (Conv)", stEmpty, stEmpty, stEmpty;
    ELSE
        
        TcpSendMessage("evaluate");
        answer := TCPReceiveMessage();      
        
        IF answer = "complete" THEN
            visionResult := 1;
            TPWrite "Vision Result: Complete";

        ELSE
            VisionResult := 2;
            TPWrite "Vision Result: Incomplete";

        ENDIF
        
        ! Close Socket connection ZZZ Change this to later moment maybe?
        !ClientCloseAndDisconnect;
        
        ! Send info to PLC according to vision result
        IF USE_PLC THEN
            SetDO QiABBVerificationStatus, visionResult-1; ! 0 = complete, 1 = incomplete  
        ENDIF
        
    ENDIF

ENDPROC

! Process the part based on the vision evaluation results
PROC ProcessPart()
       
    ! Pickup Part again and go to Home Table
    MoveL Offs(pCamera, 0, 0, 50), v200, z15, tGripper\WObj:=wobjTable;  ! Go to approach camera pos
    MoveL pCamera, v50, fine, tGripper\WObj:=wobjTable;                 ! Go to camera dropoff pos
    CloseGripper;
    MoveL Offs(pCamera, 0, 0, 50), v200, z15, tGripper\WObj:=wobjTable;  ! Go to approach camera pos
    MoveJ pHomeTable, v2500, z100, tGripper\WObj:=wobjTable;            ! Go to table middle

    ! Check vision result and process part accordingly
    IF visionResult = 1 THEN ! Part is complete, placing in onto AGV 
        TPWrite "Not yet coded";
        EXIT;   
!        MoveJ pApproachAGV, v2500, z100, tGripper\WObj:=wobjAGV;  ! Go to approach AGV pos
!        MoveL Offs(pAGV, 0, 0, 50), v200, z15, tGripper\WObj:=wobjAGV;  ! Go to approach AGV pos using offset
!        MoveL pAGV, v50, fine, tGripper\WObj:=wobjAGV;                 ! Go to AGV pickup pos
!        CloseGripper;
!        MoveL Offs(pAGV, 0, 0, 50), v200, z15, tGripper\WObj:=wobjAGV;  ! Go back to approach AGV pos using offset
!        MoveJ pApproachAGV, v2500, z100, tGripper\WObj:=wobjAGV;  ! Go to approach AGV pos
!        MoveJ pTrajectory, v2500, z100, tGripper\WObj:=wobj0;     ! Move to trajectory approach pos back to table

    ELSEIF visionResult = 2 THEN ! Part is incomplete, placing back onto conveyor
        MoveJ pTrajectory, v2500, z100, tGripper;                                   ! Move to trajectory approach pos
        MoveJ pApproachConveyor, v2500, z100, tGripper\WObj:=wobjConveyor;          ! Go to approach Conveyor pos
        MoveL Offs(pConveyor, 50, 0, 50), v200, z15, tGripper\WObj:=wobjConveyor;    ! Go to approach Conveyor pos using offset !!! Maybe add some x offset here to not drop it exactly where it was picked up
        MoveL Offs(pConveyor, 50, 0, 2), v50, fine, tGripper\WObj:=wobjConveyor;                    ! Go to Conveyor pickup pos
        OpenGripper;
        MoveL Offs(pConveyor, 50, 0, 50), v200, z15, tGripper\WObj:=wobjConveyor;    ! Go back to approach Conveyor pos using offset
        MoveJ pApproachConveyor, v2500, z100, tGripper\WObj:=wobjConveyor;          ! Go to approach Conveyor pos
        MoveJ pTrajectory, v2500, z100, tGripper\WObj:=wobj0;                       ! Move to trajectory approach pos
    ELSE 
        TPWrite "Error; Faulty vision result value";
        EXIT;   
    ENDIF

ENDPROC

PROC ResetSystem()
    MoveJ pHome, v2500, z50, tool0\WObj:=wobj0;
    WaitTime\InPos ,.1;
    ClientCloseAndDisconnect;
    Reset doValve1;
ENDPROC

PROC OpenGripper()
    SET doValve1;
    WaitTime .3;
ENDPROC

PROC CloseGripper()
    RESET doValve1;
    WaitTime .3;
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



ENDMODULE
