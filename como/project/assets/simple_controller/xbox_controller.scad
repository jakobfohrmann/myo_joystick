// -----------------------------------------------
// Simple Xbox-like controller body with side grips
// -----------------------------------------------

$fn = 64;  // smoothness

//--------------------------------------
// Rounded rectangle (controller center)
//--------------------------------------
module pill_rect(len=140, wid=80, radius=25, height=20) {
    linear_extrude(height = height)
        hull() {
            translate([ len/2 - radius,  wid/2 - radius]) circle(r = radius);
            translate([-len/2 + radius,  wid/2 - radius]) circle(r = radius);
            translate([ len/2 - radius, -wid/2 + radius]) circle(r = radius);
            translate([-len/2 + radius, -wid/2 + radius]) circle(r = radius);
        }
}

//--------------------------------------
// Controller grip (handle)
// A stretched sphere on each side
//--------------------------------------
module controller_grip(side = 1,angle =1) {
    translate([side * 65, -15, 11])
    rotate([90, 0,angle *25])    
    // side: +1 right, -1 left
        scale([0.8, 0.9, 1.8])        // shape of the grip
            sphere(r = 30);           // base sphere
}

//--------------------------------------
// Final combined controller shell
//--------------------------------------
module controller_shell() {
    union() {
        // main central body
        pill_rect(len = 130, wid = 60, radius = 25, height = 22);

        // left and right grips
        controller_grip(+1,+1);
        controller_grip(-1,-1);
    }
}

//--------------------------------------
// Render the final controller geometry
//--------------------------------------
controller_shell();

