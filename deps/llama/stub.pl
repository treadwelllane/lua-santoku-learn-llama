#!/usr/bin/perl
local $/;
open(my $fh, '<', $ARGV[0]) or die "Cannot open $ARGV[0]: $!";
my $s = <$fh>;
close $fh;

$s = "#pragma GCC diagnostic ignored \"-Wunused-parameter\"\n" . $s;

my $o = '';
my $i = 0;
while ($i < length($s)) {
    my $p = index($s, '{', $i);
    if ($p < 0) {
        $o .= substr($s, $i);
        last;
    }
    $o .= substr($s, $i, $p - $i);
    my $d = 1;
    my $j = $p + 1;
    while ($j < length($s) && $d > 0) {
        my $c = substr($s, $j, 1);
        my $cn = $j + 1 < length($s) ? substr($s, $j + 1, 1) : '';
        if ($c eq '/' && $cn eq '/') {
            my $nl = index($s, "\n", $j);
            $j = $nl < 0 ? length($s) : $nl + 1;
        } elsif ($c eq '/' && $cn eq '*') {
            my $end = index($s, "*/", $j + 2);
            $j = $end < 0 ? length($s) : $end + 2;
        } elsif ($c eq "'" ) {
            $j++;
            while ($j < length($s) && substr($s, $j, 1) ne "'") {
                $j++ if substr($s, $j, 1) eq '\\';
                $j++;
            }
            $j++;
        } elsif ($c eq '"') {
            $j++;
            while ($j < length($s) && substr($s, $j, 1) ne '"') {
                $j++ if substr($s, $j, 1) eq '\\';
                $j++;
            }
            $j++;
        } else {
            $d++ if $c eq '{';
            $d-- if $c eq '}';
            $j++;
        }
    }
    if (substr($s, $j) =~ /^\s*;/) {
        $o .= substr($s, $p, $j - $p);
    } else {
        $o .= '{ GGML_ABORT("unsupported"); }';
    }
    $i = $j;
}

open($fh, '>', $ARGV[0]) or die "Cannot write $ARGV[0]: $!";
print $fh $o;
