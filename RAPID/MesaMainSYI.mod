MODULE MesaMainSYI
    PERS wobjdata wobjTable:=[FALSE,TRUE,"",[[1324.77,-472.104,426.917],[0.999992,0.000929973,-0.000683746,0.00391044]],[[0,0,0],[1,0,0,0]]];
    PERS wobjdata wobjConveyor:=[FALSE,TRUE,"",[[225.01,1139.56,326.568],[0.999943,-0.010696,-0.000444899,0.000237186]],[[0,0,0],[1,0,0,0]]];
    PERS tooldata tGripper:=[TRUE,[[0.51636,-0.710444,275.457],[1,0,0,0]],[26,[80,0,180],[1,0,0,0],0,0,0]];
    PERS tooldata tPen:=[TRUE,[[-180.096,-2.75697,367.367],[1,0,0,0]],[22.7,[80,0,180],[1,0,0,0],0,0,0]];
    PERS robtarget pHomeTable:=[[-92.84,605.12,289.97],[0.00228675,-0.0539337,-0.998541,0.00114616],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pHome:=[[1246.17,-13.35,1540.00],[0.00170027,-0.0511174,-0.99869,-0.00183437],[-1,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pApproachCamera:=[[371.01,634.71,95.07],[0.00046576,0.0554334,0.998462,-4.78765E-05],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pCamera:=[[440.03,639.99,24.18],[0.00360889,-0.0431114,-0.999064,0.000601169],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pAway:=[[101.71,640.07,69.99],[0.00360892,-0.0554193,-0.998456,0.000647474],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pApproachConveyor:=[[329.56,101.68,179.90],[0.00508886,-0.676704,0.736203,0.00713396],[0,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pConveyor:=[[271.00,105.88,5.32],[0.00504288,-0.671369,0.741071,0.00720437],[0,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pTrajectory:=[[787.71,775.77,904.87],[0.00148502,0.494751,-0.869034,0.000424912],[0,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pTrajectory2:=[[229.77,386.33,223.91],[0.00232557,-0.055034,-0.998481,-0.000869737],[-1,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS robtarget pFinal:=[[241.85,104.43,25.70],[0.00231013,-0.0468528,-0.998899,0.00112025],[-1,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]]; 
    PERS intnum visionResult := 0;
    
    ! Configuration constants
    CONST bool USE_PLC := False;
    CONST bool USE_VISION := True;

    ! Main routine
    PROC main()
        TPErase;                ! Clear TP output
        SetupPlcCom;            ! Setup PLC com.
        SetupVisionSystem;      ! Setup Vision system
        GetPart;                ! Pickup Part and place to Camera System
        VisionEvaluation;       ! Evaluate Part using Vision Computer
        ProcessPart;            ! Move Part away
        ResetSystem;            ! Reset whole system, home robot
    ENDPROC
    
    
PROC SetupPlcCom()
    ! Update status
    SetGO QiABBStatus, 1;              ! Status to Init
    SetDO QiABBVerificationStatus, 1;  ! Reset part status to not good
            
    ! Log output
    IF USE_PLC THEN
        TPWrite "PLC Communication ON";
    ELSE
        TPWrite "PLC Communication OFF";
    ENDIF
    
ENDPROC

PROC SetupVisionSystem()
    VAR string receivedMessage;
    
    IF USE_VISION THEN
        TPWrite "Vision System ON";

        !Create and Connect TCP Client to Server
        ClientCreateAndConnect;
    
        !Wait for Server message
        receivedMessage:=TCPReceiveMessage();
        TPWrite("Server message: "+receivedMessage);
        waitTime(.25);
    
        !Respond to Server
        TCPSendMessage("robot ready");
        
    ELSE
        TPWrite "Vision System OFF";
    ENDIF
ENDPROC

PROC GetPart()   
    ! Update status
    SetGO QiABBStatus, 2;   ! Status to Get Part

    ! Define where to get the part from and wait for Execution signal
    IF USE_PLC THEN
        WaitDI IxABBStart, 1;
    ENDIF
    
    ! Make sure Robot is at home position and the gripper is open
    MoveJ pHome, v2500, z50, tool0\WObj:=wobj0;
    OpenGripper;
    
    ! Pickup at Conveyor
    PickupAtConveyor(True);
     
    ! Place part below Camera and place it
    MoveJ pHomeTable, v2500, z100, tGripper\WObj:=wobjTable;            ! Go to table middle
    MoveJ Offs(pCamera, 0, 0, 50), v200, z15, tGripper\WObj:=wobjTable;  ! Go to approach camera pos
    MoveL pCamera, v50, fine, tGripper\WObj:=wobjTable;                 ! Go to camera dropoff pos
    OpenGripper;
    MoveL Offs(pCamera, 0, 0, 50), v200, z15, tGripper\WObj:=wobjTable;  ! Go to approach camera pos
    
    ! Move robot out of camera frame
    MoveL pAway, v2500, z50, tGripper\WObj:=wobjTable;              ! Go to trajectory approach pos for Camera
    WaitTime\InPos ,.1;
        
ENDPROC

! Send request to Vision PC to evaluate the part
PROC VisionEvaluation()
    VAR string answer;
    
    ! Update status
    SetGO QiABBStatus, 3;   ! Status to part eval.
    
    IF NOT USE_VISION THEN
        ! Manual vision system override
        TPReadFK visionResult, "Choose vision result?", "Complete (AGV)", "Incomplete (Conveyor)", stEmpty, stEmpty, stEmpty;
        VisionResult := visionResult -1;
    ELSE
        ! Vision system evaluation
        TcpSendMessage("evaluate");
        answer := TCPReceiveMessage();      
        
        IF answer = "COMPLETE" THEN
            visionResult := 0;
            TPWrite "Vision Result: Complete";

        ELSE
            VisionResult := 1;
            TPWrite "Vision Result: Incomplete";

        ENDIF
               
        ! Send info to PLC according to vision result
        IF USE_PLC THEN
            SetDO QiABBVerificationStatus, visionResult; ! 0 = complete, 1 = incomplete  
        ENDIF
        
    ENDIF

ENDPROC

! Process the part based on the vision evaluation results
PROC ProcessPart()
    
    ! Update status
    SetGO QiABBStatus, 4;   ! Status to process part
       
    ! Pickup Part again and go to Home Table
    MoveL Offs(pCamera, 0, 0, 50), v200, z15, tGripper\WObj:=wobjTable;  ! Go to approach camera pos
    MoveL pCamera, v50, fine, tGripper\WObj:=wobjTable;                 ! Go to camera dropoff pos
    CloseGripper;
    MoveL Offs(pCamera, 0, 0, 50), v200, z15, tGripper\WObj:=wobjTable;  ! Go to approach camera pos

    ! Check vision result and process part accordingly
    IF visionResult = 0 THEN ! Part is complete, placing onto Table
        MoveJ pTrajectory2, v2500, z100, tGripper\WObj:=wobjTable;  ! Go to approach Table Storage pos
        MoveL Offs(pFinal, 200, 0, 40), v2500, z15, tGripper\WObj:=wobjTable;  ! Go to approach Table Storage pos using offset
        MoveL Offs(pFinal, 200, 0, 10), v200, z15, tGripper\WObj:=wobjTable;  ! Go to approach Table Storage pos using offset
        MoveL Offs(pFinal, 0, 0, 10), v200, z15, tGripper\WObj:=wobjTable;  ! Go to approach Table Storage pos using offset
        MoveL pFinal, v50, fine, tGripper\WObj:=wobjTable;                 ! Go to Table Storage pickup pos
        OpenGripper;
        MoveL Offs(pFinal, 0, 0, 40), v200, z15, tGripper\WObj:=wobjTable;  ! Go back to approach Table Storage pos using offset
    ELSEIF visionResult = 1 THEN ! Part is incomplete, placing back onto conveyor
        PickupAtConveyor(False);
    ELSE 
        TPWrite "Error; Faulty vision result value";
        EXIT;   
    ENDIF

ENDPROC

PROC ResetSystem()
    ! Update status
    SetGO QiABBStatus, 5;   ! Status to Reset system
    MoveJ pHome, v2500, z50, tool0\WObj:=wobj0;
    WaitTime\InPos ,.1;
    ClientCloseAndDisconnect;
    Reset doValve1;
    
    ! Update status
    SetGO QiABBStatus, 0;   ! Status to stopped system   
    
ENDPROC

PROC OpenGripper()
    SET doValve1;
    WaitTime .3;
ENDPROC

PROC CloseGripper()
    RESET doValve1;
    WaitTime .3;
ENDPROC

PROC PickupAtConveyor(bool mode)
        MoveJ pTrajectory, v2500, z100, tGripper;                       ! Move to trajectory approach pos back to table
        MoveL Offs(pConveyor, 0, 0, 75), v200, z20, tGripper\WObj:=wobjConveyor;  ! Go to approach Conveyor pos using offset
        MoveL pConveyor, v50, fine, tGripper\WObj:=wobjConveyor;                 ! Go to Conveyor pickup pos
        IF mode THEN
            CloseGripper;
        ELSE
            OpenGripper;
        ENDIF
        MoveL Offs(pConveyor, 0, 0, 75), v200, z20, tGripper\WObj:=wobjConveyor;  ! Go back to approach Conveyor pos using offset
        MoveJ pTrajectory, v2500, z100, tGripper;                       ! Move to trajectory approach pos back to table
ENDPROC



ENDMODULE
