extends SceneTree
## UC14 validace (L40S sandbox): dávka kopů přes čisté core skripty
## (ServeGenerator → BallPhysics → ContactModel) se stejnou logikou jako
## game_state.gd (pre-simulace ideálu, lokální rámec kopu, průlet rovinou
## brány, NEDOLETĚL po zastavení). Vypíše trajektorie do JSON pro srovnání
## s JAX portem (tmobile-sandbox/uc14/volley_jax.py). Není to test —
## jméno nekončí na _test.gd, run_all.sh ho nespouští.
##
##   godot --headless --path . --script res://tests/tools/kick_dump.gd -- in.json out.json
##
## in.json: [{"seed": int, "ox": f, "oy": f, "curve": f, "power": f, "dtick": int}, …]
## dtick = posun puštění vůči ideálnímu ticku (timing_error = dtick/120 s).

const T := preload("res://scripts/util/tuning.gd")
const BallScript := preload("res://scripts/core/ball_physics.gd")
const ServeGen := preload("res://scripts/core/serve_generator.gd")
const CM := preload("res://scripts/core/contact_model.gd")

const DT := 1.0 / 120.0
const MAX_FLIGHT_TICKS := 120 * 12

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() < 2:
		push_error("použití: -- in.json out.json")
		quit(2)
		return
	var kicks: Array = JSON.parse_string(FileAccess.get_file_as_string(args[0]))
	var out: Array = []
	for k in kicks:
		out.append(_run_kick(k))
	var f := FileAccess.open(args[1], FileAccess.WRITE)
	f.store_string(JSON.stringify(out, "", false, true))
	f.close()
	print("kick_dump: %d kopů → %s" % [out.size(), args[1]])
	quit(0)

func _v(v: Vector3) -> Array:
	return [v.x, v.y, v.z]

## Kopie game_state._presim_ideal_time (vrací čas i počet ticků).
func _presim_ideal(s: Dictionary) -> Array:
	var scratch: BallPhysics = BallScript.new()
	scratch.position = s.pos
	scratch.launch(s.vel, s.omega)
	var t := 0.0
	var ticks := 0
	var apex_t := 0.0
	var apex_ticks := 0
	var apex_h := 0.0
	while t < 8.0 and scratch.flying and scratch.bounce_count < 2:
		scratch._physics_process(DT)
		t += DT
		ticks += 1
		if scratch.bounce_count == 1:
			if scratch.position.y > apex_h:
				apex_h = scratch.position.y
				apex_t = t
				apex_ticks = ticks
			if scratch.vel.y < 0.0 and scratch.position.y <= T.IDEAL_KICK_HEIGHT:
				scratch.free()
				return [t, ticks]
	scratch.free()
	return [apex_t, apex_ticks] if apex_t > 0.0 else [t, ticks]

func _run_kick(k: Dictionary) -> Dictionary:
	var gen: ServeGenerator = ServeGen.new(int(k.seed))
	var s: Dictionary = gen.next_serve()
	var ideal: Array = _presim_ideal(s)
	var res := {"seed": int(k.seed), "serve_pos": _v(s.pos), "serve_vel": _v(s.vel),
			"serve_omega": _v(s.omega), "t_ideal": ideal[0], "ideal_tick": ideal[1]}

	var ball: BallPhysics = BallScript.new()
	ball.position = s.pos
	ball.launch(s.vel, s.omega)
	var kick_tick: int = int(ideal[1]) + int(k.dtick)
	var serve_time := 0.0
	for i in kick_tick:
		ball._physics_process(DT)
		serve_time += DT
	res["kick_tick"] = kick_tick
	res["kick_pos"] = _v(ball.position)
	res["kick_vel"] = _v(ball.vel)
	if kick_tick < 0 or ball.bounce_count >= 2 or not ball.flying:
		res["outcome"] = "LATE"   # druhý dopad = šance pryč
		ball.free()
		return res

	var err: float = serve_time - float(ideal[0])
	var fwd := Vector3(0, 0, -T.GOAL_DISTANCE) - ball.position
	fwd.y = 0.0
	fwd = fwd.normalized()
	var right := fwd.cross(Vector3.UP)
	var ball_vel_local := Vector3(ball.vel.dot(right), ball.vel.y, -ball.vel.dot(fwd))
	var c: Dictionary = CM.compute_kick(Vector2(k.ox, k.oy), float(k.curve),
			float(k.power), err, ball_vel_local)
	res["err"] = err
	res["rating"] = int(c.rating)
	if not c.hit:
		res["outcome"] = "WHIFF"
		ball.free()
		return res
	var vel: Vector3 = right * c.vel.x + Vector3.UP * c.vel.y - fwd * c.vel.z
	var omega: Vector3 = right * c.omega.x + Vector3.UP * c.omega.y - fwd * c.omega.z
	res["launch_vel"] = _v(vel)
	res["launch_omega"] = _v(omega)
	ball.launch(vel, omega)

	# let: stejné pořadí jako game_state (_check_goal_crossing na dvojici
	# po sobě jdoucích pozic, zastavení míče = NEDOLETĚL)
	var plane_z := -T.GOAL_DISTANCE
	var traj: Array = []
	var outcome := "SHORT"
	for i in MAX_FLIGHT_TICKS:
		var prev_z := ball.position.z
		ball._physics_process(DT)
		traj.append_array(_v(ball.position))
		if not ball.flying:
			outcome = "SHORT"
			break
		if prev_z > plane_z and ball.position.z <= plane_z:
			var t := (plane_z - prev_z) / (ball.position.z - prev_z)
			var cross_x := lerpf(ball.position.x - ball.vel.x * DT, ball.position.x, t)
			var cross_y := lerpf(ball.position.y - ball.vel.y * DT, ball.position.y, t)
			res["cross"] = [cross_x, cross_y]
			res["cross_speed"] = ball.vel.length()
			outcome = "GOAL" if (absf(cross_x) < T.GOAL_HALF_WIDTH
					and cross_y < T.GOAL_HEIGHT) else "WIDE"
			break
	res["outcome"] = outcome
	res["flight_ticks"] = traj.size() / 3
	res["traj"] = traj
	ball.free()
	return res
