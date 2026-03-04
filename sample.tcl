#
# For generality: load Tk
#

package require Tk

#
# Define the callback procedure used by the pushbutton
#
proc handleMsg {} {
    tk_messageBox -title Message -message $::msg -type ok
    set ::msg ""
}

#
# Create the widgets
#

label  .label -text "Enter text:"
entry  .entry -textvariable msg
button .button -text Run -command handleMsg

#
# Make the widgets visible
#

grid .label  .entry -sticky news
grid .button -

#
# We want to be able to resize the entry ...
#
grid columnconfigure . 1 -weight 1
