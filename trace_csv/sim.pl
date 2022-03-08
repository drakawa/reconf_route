#!/usr/bin/perl

@te_rate_list = ( 2 );
$trace = "bt.w.64";
$sample_period = 10000;
$base_config = "../template.cfg";
$local_config = "local.cfg";

foreach ( @te_rate_list ) {
        $te_rate = $_;
        $base_name  = "${trace}.tr";
        $expanded_trace = "tmp.${_}.tr";

        # Time expansion
        unlink($expanded_trace);
        $clock = &time_expansion($te_rate, $base_name, $expanded_trace);
        $max_samples = int($clock / $sample_period) + 1;
        printf("Time Expansion Rate %f (max_samples %d) ************* \n",
                $te_rate, $max_samples);

        # Configuration file
        unlink($local_config);
        system("cat $base_config | sed s/__TRACE__/$expanded_trace/ | sed s/__PERIOD__/$max_samples/ > $local_config");

        # Simulation!
        system("../../src/booksim local.cfg | grep -e \"^Overall\" -e \"Average hops\" -e \"^Info:\"");

        # Please delete unnecessary temporary tarce files!
        unlink($expanded_trace);
}

sub time_expansion {
        my($rate)       = $_[0];
        my($fn_in)      = $_[1];
        my($fn_out)     = $_[2];
        my($line, $i, @data, $clock);
        if ( ! -e $fn_in ) {
                printf("Error: file not found (%s). \n", $fn_in);
                die;
        }
        open(fi, "<$fn_in");
        open(fo, ">$fn_out");
        $i = 0;
        while ( $line = <fi> ) {
                if ( $line =~ /^#/ || $i == 0 ) {
                        printf(fo "%s", $line);
                        $i++;
                        next;
                }
                chomp($line);
                $line =~ tr/ //s;
                if ( $line =~ m/^\s+(.*)/) {
                        $line = $1;
                }
                @data = split(/\s+/, $line);
                $clock = int($data[0] * $rate);
                #printf(fo "%d %d %d %d\n", $clock,$data[1],$data[2],$data[3]);
                printf(fo "%d %d %d %d\n", $clock, $data[1], $data[2], 10);
                $i++;
        }
        close(fi);
        close(fo);
        return $clock;
}
